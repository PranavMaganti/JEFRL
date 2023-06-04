from pathlib import Path
import pickle
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import random_split
import tqdm
from transformers import get_linear_schedule_with_warmup
from transformers import RobertaConfig
from transformers import RobertaForMaskedLM


with open("ASTBERTa/vocab_data.pkl", "rb") as f:
    vocab_data = pickle.load(f)

with open("ASTBERTa/data.pkl", "rb") as f:
    data = pickle.load(f)


PAD_TOKEN = "<pad>"
CLS_TOKEN = "<s>"
SEP_TOKEN = "</s>"
MASK_TOKEN = "<mask>"
UNK_TOKEN = "<unk>"

special_tokens = [PAD_TOKEN, CLS_TOKEN, MASK_TOKEN, SEP_TOKEN, UNK_TOKEN]

token_to_id = vocab_data["token_to_id"]
vocab = vocab_data["vocab"]


class FragDataset(Dataset[list[int]]):
    def __init__(self, data: list[list[int]]):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int) -> list[int]:
        return self.data[index]


MAX_SEQ_LEN = 512
MLM_PROB = 0.15
MODEL_SAVE_PATH = "ASTBERTa/models/sub-sequence/"


def seq_data_collator(batch: list[list[int]]) -> dict[str, torch.Tensor]:
    seqs: list[torch.Tensor] = []

    for x in batch:
        if torch.rand(1).item() < 0.75:
            random_start_idx = torch.randint(low=2, high=len(x), size=(1,)).item()
            seq = [token_to_id[CLS_TOKEN]] + x[
                random_start_idx : random_start_idx + MAX_SEQ_LEN - 1
            ]
        else:
            seq = x[:MAX_SEQ_LEN]

        assert len(seq) <= MAX_SEQ_LEN
        seqs.append(torch.tensor(seq))

    inputs = torch.nn.utils.rnn.pad_sequence(seqs, batch_first=True)

    labels = inputs.clone()

    special_token_mask = torch.zeros_like(labels).float()
    special_token_mask[(labels >= 0) & (labels <= len(special_tokens))] = 1.0
    special_token_mask = special_token_mask.bool()

    probability_matrix = torch.full(labels.shape, MLM_PROB)
    probability_matrix.masked_fill_(special_token_mask, value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = (
        torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    )
    inputs[indices_replaced] = token_to_id[MASK_TOKEN]
    labels[~masked_indices] = -100

    # 10% of the time, we replace masked input tokens with random word
    indices_random = (
        torch.bernoulli(torch.full(labels.shape, 0.5)).bool()
        & masked_indices
        & ~indices_replaced
    )
    random_words = torch.randint(len(vocab), labels.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]

    attention_mask = torch.ones_like(inputs, dtype=torch.float)
    attention_mask[inputs == token_to_id[PAD_TOKEN]] = 0.0

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return {
        "input_ids": inputs,
        "labels": labels,
        "attention_mask": attention_mask,
    }


def evaluate_batch(model: RobertaForMaskedLM, batch: dict[str, Any]) -> torch.Tensor:
    criterion = nn.CrossEntropyLoss()

    input_ids = batch["input_ids"].to(device)
    labels = batch["labels"].to(device)
    attention_mask = batch["attention_mask"].to(device)

    out = model(input_ids, attention_mask=attention_mask).logits
    return criterion(out.view(-1, vocab_size), labels.view(-1))


def evaluate(model: RobertaForMaskedLM, val_loader: DataLoader[list[int]]):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in tqdm.tqdm(val_loader):
            loss = evaluate_batch(model, batch)
            total_loss += loss.item()

    return total_loss / len(val_loader)


def train(
    model: RobertaForMaskedLM,
    train_loader: DataLoader[list[int]],
    val_loader: DataLoader[list[int]],
    optim: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler.LambdaLR,
    model_save_path: Path = Path(MODEL_SAVE_PATH),
    epochs: int = 8,
):
    steps = 0

    train_losses = []
    val_losses = []

    for epoch in (pbar := tqdm.trange(epochs)):
        model.train()
        epoch_loss = 0
        per_batch_loss = []
        for _, batch in (
            ibar := tqdm.tqdm(
                enumerate(train_loader), leave=True, total=len(train_loader)
            )
        ):
            loss = evaluate_batch(model, batch)
            loss.backward()
            lr_scheduler.step()
            optim.step()
            optim.zero_grad()

            steps += 1
            epoch_loss += loss.item()
            per_batch_loss.append(loss.item())
            # print(f"Epoch: {epoch}, Batch: {i}, Loss: {loss.item()}")
            ibar.set_postfix({"loss": loss.item()})

            if steps % 500 == 0:
                torch.save(model, model_save_path / f"model_{steps}.pt")

        val_loss = evaluate(model, val_loader)
        pbar.set_postfix(
            {"val_loss": val_loss, "train_loss": epoch_loss / len(train_loader)}
        )
        print(
            f"Epoch: {epoch}, Val Loss: {val_loss}, Train Loss: {epoch_loss / len(train_loader)}"
        )

        train_losses.append(per_batch_loss)
        val_losses.append(val_loss)

        pickle.dump(train_losses, open(model_save_path / "train_losses.pkl", "wb"))
        pickle.dump(val_losses, open(model_save_path / "val_losses.pkl", "wb"))


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vocab_size = len(vocab)  # size of vocabulary
intermediate_size = 3072  # embedding dimension
hidden_size = 768

num_hidden_layers = 6
num_attention_heads = 12
dropout = 0.1

batch_size = 64

dataset = FragDataset(data)
train_split, val_split, test_split = random_split(dataset, [0.8, 0.1, 0.1])

train_loader = DataLoader(
    train_split, batch_size=batch_size, shuffle=True, collate_fn=seq_data_collator
)
val_loader = DataLoader(
    val_split, batch_size=batch_size, shuffle=True, collate_fn=seq_data_collator
)
test_loader = DataLoader(
    test_split, batch_size=batch_size, shuffle=True, collate_fn=seq_data_collator
)
config = RobertaConfig(
    vocab_size=vocab_size,
    hidden_size=hidden_size,
    num_hidden_layers=num_hidden_layers,
    num_attention_heads=num_attention_heads,
    intermediate_size=intermediate_size,
    hidden_dropout_prob=dropout,
    max_position_embeddings=MAX_SEQ_LEN + 2,
)
model = RobertaForMaskedLM(config).to(device)
model = torch.nn.DataParallel(model, device_ids=[0, 1])

optim = torch.optim.AdamW(
    model.parameters(),
    lr=6e-4,
    eps=1e-6,
    weight_decay=0.01,
    betas=(0.9, 0.98),
)
lr_scheduler = get_linear_schedule_with_warmup(
    optimizer=optim, num_warmup_steps=2400, num_training_steps=50000
)

print(
    f"The model has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters"
)
print(model)

train(model, train_loader, val_loader, optim, lr_scheduler, epochs=100)
