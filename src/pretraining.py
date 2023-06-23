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
from transformers import RobertaConfig
from transformers import RobertaForMaskedLM


os.makedirs("out/pretraining", exist_ok=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print("Loading data...")
with open("ASTBERTa/vocab_data.pkl", "rb") as f:
    vocab_data = pickle.load(f)

with open("data/pretraining_data.pkl", "rb") as f:
    data = pickle.load(f)


MAX_SEQ_LEN = 512
MLM_PROB = 0.15

PAD_TOKEN = "<pad>"
CLS_TOKEN = "<s>"
SEP_TOKEN = "</s>"
MASK_TOKEN = "<mask>"
UNK_TOKEN = "<unk>"

special_tokens = [PAD_TOKEN, CLS_TOKEN, MASK_TOKEN, SEP_TOKEN, UNK_TOKEN]

token_to_id = vocab_data["token_to_id"]
vocab = vocab_data["vocab"]


class ASTFragDataset(Dataset[list[int]]):
    def __init__(self, data: list[list[int]]):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int) -> list[int]:
        return self.data[index]


start = datetime.now()
save_folder_name = start.strftime("%Y-%m-%dT%H:%M:.%f")
save_folder = Path(f"out/pretraining/{save_folder_name}")
os.makedirs(save_folder, exist_ok=True)


def seq_data_collator(batch: list[list[int]]) -> dict[str, torch.Tensor]:
    seqs: list[torch.Tensor] = []

    for x in batch:
        max_len = min(MAX_SEQ_LEN, len(x))

        if random.random() < 0.75:
            random_start_idx = random.randint(1, len(x) - 1)
            random_len = random.randint(1, max_len - 1)
            seq = [token_to_id[CLS_TOKEN]] + x[
                random_start_idx : random_start_idx + random_len
            ]
        else:
            seq = x[:max_len]

        assert len(seq) <= MAX_SEQ_LEN
        seqs.append(torch.tensor(seq, device=device, dtype=torch.long))

    inputs = torch.nn.utils.rnn.pad_sequence(seqs, batch_first=True)
    labels = inputs.clone()

    special_token_mask = torch.zeros_like(labels, device=device).float()
    special_token_mask[(labels >= 0) & (labels <= len(special_tokens))] = 1.0
    special_token_mask = special_token_mask.bool()

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

    # 10% of the time, we replace masked input tokens with random word
    indices_random = (
        torch.bernoulli(torch.full(labels.shape, 0.5, device=device)).bool()
        & masked_indices
        & ~indices_replaced
    )
    random_words = torch.randint(
        len(vocab), labels.shape, dtype=torch.long, device=device
    )
    inputs[indices_random] = random_words[indices_random]

    attention_mask = torch.ones_like(inputs, dtype=torch.float, device=device)
    attention_mask[inputs == token_to_id[PAD_TOKEN]] = 0.0

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return {
        "input_ids": inputs,
        "labels": labels,
        "attention_mask": attention_mask,
    }


vocab_size = len(vocab)  # size of vocabulary
intermediate_size = 2048  # embedding dimension
hidden_size = 512

num_hidden_layers = 3
num_attention_heads = 8
dropout = 0

batch_size = 64

learning_rate = 5e-5


dataset = ASTFragDataset(data)
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

print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")


optim = torch.optim.AdamW(
    model.parameters(),
    lr=learning_rate,
    eps=1e-6,
    weight_decay=0.01,
    betas=(0.9, 0.98),
)

EPOCHS = 50
steps = 0
train_losses = []
val_losses = []

with open(save_folder / "hyperparameters.json", "w") as f:
    f.write(
        json.dumps(
            {
                "test_split": list(test_split.indices),
                "val_split": list(val_split.indices),
                "LR": learning_rate,
                "dropout": dropout,
            }
        )
    )


def evaluate(model: RobertaForMaskedLM, val_loader: DataLoader[list[int]]):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in tqdm.tqdm(val_loader):
            loss = model(**batch).loss
            total_loss += loss.item()

    return total_loss / len(val_loader)


for epoch in (pbar := tqdm.trange(EPOCHS)):
    model.train()
    epoch_loss = 0
    per_batch_loss = []
    for _, batch in (
        ibar := tqdm.tqdm(enumerate(train_loader), leave=True, total=len(train_loader))
    ):
        loss = model(**batch).loss
        optim.zero_grad()
        loss.backward()
        optim.step()

        steps += 1
        epoch_loss += loss.item()
        per_batch_loss.append(loss.item())
        # print(f"Epoch: {epoch}, Batch: {i}, Loss: {loss.item()}")
        ibar.set_postfix({"loss": loss.item()})

        if steps % 500 == 0:
            torch.save(model, save_folder / f"model_{steps}.pt")

    val_loss = evaluate(model, val_loader)
    print(
        f"Epoch: {epoch}, Val Loss: {val_loss}, Train Loss: {epoch_loss / len(train_loader)}"
    )

    train_losses.append(per_batch_loss)
    val_losses.append(val_loss)

    pickle.dump(train_losses, open(save_folder / "train_losses.pkl", "wb"))
    pickle.dump(val_losses, open(save_folder / "val_losses.pkl", "wb"))
