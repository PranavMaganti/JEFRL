import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import pickle
import random

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import random_split
import tqdm
from transformer.ast_transformer import get_ast_transformer_config
from transformer.special_tokens import CLS_TOKEN
from transformer.special_tokens import MASK_TOKEN
from transformer.special_tokens import PAD_TOKEN
from transformer.special_tokens import special_tokens
from transformers import get_linear_schedule_with_warmup
from transformers import RobertaForMaskedLM

from utils.device import get_device


parser = argparse.ArgumentParser(
    prog="JEFRL Pre-training", description="AST Transformer Pre-training for JEFRL"
)

parser.add_argument(
    "--data-dir",
    type=Path,
    default=Path("data/"),
    help="The path to which the vocabulary and pretraining data will be saved",
)
parser.add_argument(
    "--output",
    type=Path,
    default=Path("out/pretraining/"),
    help="The path to the output directory in which a new folder with the pretraining results will be saved",
)
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--batch-size", type=int, default=64)
# Best value from the grid search in the report (Table 6.2)
parser.add_argument("--lr", type=float, default=5e-4)
parser.add_argument("--warmup-ratio", type=float, default=0.06)
args = parser.parse_args()

if not args.data_dir.exists():
    raise ValueError(f"Data directory {args.data_dir} does not exist")

os.makedirs(args.output, exist_ok=True)


device = get_device()
print(f"Using device {device}")

print("Loading data...")
with open(args.data_dir / "vocab_data.pkl", "rb") as f:
    vocab_data = pickle.load(f)

with open(args.data_dir / "frag_data.pkl", "rb") as f:
    frag_data = pickle.load(f)


MAX_SEQ_LEN = 512
MLM_PROB = 0.15

token_to_id = vocab_data["token_to_id"]
vocab = vocab_data["vocab"]
num_special_tokens = len(special_tokens)
# Special tokens occupy the first ids of the vocabulary
assert all(token_to_id[t] < num_special_tokens for t in special_tokens)

# Sequences need at least one fragment after the CLS token
frag_seqs = [seq for seq in frag_data["frag_data"] if len(seq) >= 2]


class ASTFragDataset(Dataset[list[int]]):
    def __init__(self, data: list[list[int]]):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int) -> list[int]:
        return self.data[index]


start = datetime.now()
save_folder = args.output / start.strftime("%Y-%m-%dT%H-%M-%S")
os.makedirs(save_folder, exist_ok=True)


def seq_data_collator(batch: list[list[int]]) -> dict[str, torch.Tensor]:
    seqs: list[torch.Tensor] = []

    for x in batch:
        max_len = min(MAX_SEQ_LEN, len(x))

        if random.random() < 0.75:
            # Random sub-sequence, as the agent sees arbitrary subtrees of programs
            random_start_idx = random.randint(1, len(x) - 1)
            random_len = random.randint(1, max_len - 1)
            seq = [token_to_id[CLS_TOKEN]] + x[
                random_start_idx : random_start_idx + random_len
            ]
        else:
            seq = x[:max_len]

        assert len(seq) <= MAX_SEQ_LEN
        seqs.append(torch.tensor(seq, dtype=torch.long))

    inputs = torch.nn.utils.rnn.pad_sequence(
        seqs, batch_first=True, padding_value=token_to_id[PAD_TOKEN]
    ).to(device)
    labels = inputs.clone()

    special_token_mask = labels < num_special_tokens

    probability_matrix = torch.full(labels.shape, MLM_PROB, device=device)
    probability_matrix.masked_fill_(special_token_mask, value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = (
        torch.bernoulli(torch.full(labels.shape, 0.8, device=device)).bool()
        & masked_indices
    )
    inputs[indices_replaced] = token_to_id[MASK_TOKEN]
    labels[~masked_indices] = -100

    # 10% of the time, we replace masked input tokens with a random (non-special) token
    indices_random = (
        torch.bernoulli(torch.full(labels.shape, 0.5, device=device)).bool()
        & masked_indices
        & ~indices_replaced
    )
    random_words = torch.randint(
        num_special_tokens, len(vocab), labels.shape, dtype=torch.long, device=device
    )
    inputs[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    attention_mask = (inputs != token_to_id[PAD_TOKEN]).long()

    return {
        "input_ids": inputs,
        "labels": labels,
        "attention_mask": attention_mask,
    }


dataset = ASTFragDataset(frag_seqs)
train_split, val_split, test_split = random_split(
    dataset, [0.8, 0.1, 0.1], generator=torch.Generator().manual_seed(0)
)

train_loader = DataLoader(
    train_split, batch_size=args.batch_size, shuffle=True, collate_fn=seq_data_collator
)
val_loader = DataLoader(
    val_split, batch_size=args.batch_size, shuffle=False, collate_fn=seq_data_collator
)
test_loader = DataLoader(
    test_split, batch_size=args.batch_size, shuffle=False, collate_fn=seq_data_collator
)

config = get_ast_transformer_config(len(vocab), token_to_id)
model = RobertaForMaskedLM(config).to(device)

print(model)
print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

optim = torch.optim.AdamW(
    model.parameters(),
    lr=args.lr,
    eps=1e-6,
    weight_decay=0.01,
    betas=(0.9, 0.98),
)
total_steps = args.epochs * len(train_loader)
scheduler = get_linear_schedule_with_warmup(
    optim, int(args.warmup_ratio * total_steps), total_steps
)

steps = 0
train_losses = []
val_losses = []

with open(save_folder / "hyperparameters.json", "w") as f:
    json.dump(
        {
            "test_split": list(test_split.indices),
            "val_split": list(val_split.indices),
            "LR": args.lr,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "dropout": config.hidden_dropout_prob,
            "device": str(device),
        },
        f,
    )


def mlm_loss(model: RobertaForMaskedLM, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    """Masked language modelling loss, applying the vocabulary head only at the
    masked positions. RobertaForMaskedLM would compute scores over the whole
    vocabulary at every position (batch x 512 x vocab floats, several GB), even
    though only the ~15% masked positions contribute to the loss."""
    hidden = model.roberta(
        input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
    ).last_hidden_state
    labels = batch["labels"]
    masked = labels != -100
    logits = model.lm_head(hidden[masked])
    return torch.nn.functional.cross_entropy(logits, labels[masked])


def evaluate(model: RobertaForMaskedLM, loader: DataLoader[list[int]]):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in tqdm.tqdm(loader, leave=False):
            loss = mlm_loss(model, batch)
            total_loss += loss.item()

    return total_loss / len(loader)


best_val_loss = float("inf")
for epoch in (pbar := tqdm.trange(args.epochs)):
    model.train()
    epoch_loss = 0
    per_batch_loss = []
    for _, batch in (
        ibar := tqdm.tqdm(enumerate(train_loader), leave=False, total=len(train_loader))
    ):
        loss = mlm_loss(model, batch)
        optim.zero_grad()
        loss.backward()
        optim.step()
        scheduler.step()

        steps += 1
        epoch_loss += loss.item()
        per_batch_loss.append(loss.item())
        ibar.set_postfix({"loss": loss.item()})

    val_loss = evaluate(model, val_loader)
    print(
        f"Epoch: {epoch}, Val Loss: {val_loss}, Train Loss: {epoch_loss / len(train_loader)}"
    )

    # State dicts rather than pickled model objects, loadable by get_ast_transformer_model
    torch.save(model.state_dict(), save_folder / f"model_{steps}.pt")
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), save_folder / "model_best.pt")

    train_losses.append(per_batch_loss)
    val_losses.append(val_loss)

    with open(save_folder / "train_losses.pkl", "wb") as f:
        pickle.dump(train_losses, f)
    with open(save_folder / "val_losses.pkl", "wb") as f:
        pickle.dump(val_losses, f)

model.load_state_dict(torch.load(save_folder / "model_best.pt", map_location=device))
print(f"Test loss of best model: {evaluate(model, test_loader)}")
print(f"Best model saved to {save_folder / 'model_best.pt'}")
