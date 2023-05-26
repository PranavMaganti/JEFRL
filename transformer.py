import math
import pickle

import torch
from torch import Tensor, nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import RobertaForMaskedLM, RobertaConfig
import tqdm

with open("ASTBERTa/vocab.pkl", "rb") as f:
    vocab = pickle.load(f)

with open("ASTBERTa/frag_to_id.pkl", "rb") as f:
    frag_to_id = pickle.load(f)

with open("ASTBERTa/data.pkl", "rb") as f:
    data = pickle.load(f)


PAD_TOKEN = "<PAD>"
CLS_TOKEN = "<CLS>"
SOS_TOKEN = "<SOS>"
EOS_TOKEN = "<EOS>"
UNK_TOKEN = "<UNK>"
MASK_TOKEN = "<MASK>"

special_tokens = [PAD_TOKEN, CLS_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN, MASK_TOKEN]


class FragDataset(Dataset[list[int]]):
    def __init__(self, data: list[list[int]]):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int) -> list[int]:
        return self.data[index]


MAX_SEQ_LEN = 1024
MLM_PROB = 0.15


def seq_data_collator(batch: list[list[int]]) -> dict[str, torch.Tensor]:
    max_len = min(max(map(lambda x: len(x), batch)), MAX_SEQ_LEN)

    padded_seqs: list[list[int]] = []
    for seq in batch:
        if len(seq) > max_len:
            seq = seq[:max_len]
        padded_seq = seq + [frag_to_id[PAD_TOKEN]] * (max_len - len(seq))
        padded_seqs.append(padded_seq)

    inputs = torch.tensor(padded_seqs, dtype=torch.long)
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
    inputs[indices_replaced] = frag_to_id[MASK_TOKEN]
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
    attention_mask[inputs == frag_to_id[PAD_TOKEN]] = 0.0

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return {
        "input_ids": inputs,
        "labels": labels,
        "attention_mask": attention_mask,
    }


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vocab_size = len(vocab)  # size of vocabulary
intermediate_size = 1200  # embedding dimension
hidden_size = (
    300  # dimension of the feedforward network model in ``nn.TransformerEncoder``
)

num_hidden_layers = (
    12  # number of ``nn.TransformerEncoderLayer`` in ``nn.TransformerEncoder``
)
num_attention_heads = 12  # number of heads in ``nn.MultiheadAttention``
dropout = 0.1  # dropout probability

batch_size = 16

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
    max_position_embeddings=MAX_SEQ_LEN,
)
model = RobertaForMaskedLM(config)

print(
    f"The model has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters"
)
print(model)


def evaluate(model: RobertaForMaskedLM, val_loader: DataLoader[list[int]]):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            out = model(input_ids, attention_mask=attention_mask)
            N, d, C = out.shape
            labels = batch["labels"].T.to(device)
            loss = criterion(out.view(N, C, d), labels)
            total_loss += loss.item()

    return total_loss / len(val_loader)


def train(
    model: RobertaForMaskedLM,
    train_loader: DataLoader[list[int]],
    val_loader: DataLoader[list[int]],
):
    model.train()
    epochs = 500
    criterion = nn.CrossEntropyLoss()
    optim = torch.optim.AdamW(
        model.parameters(), lr=6e-4, eps=1e-6, weight_decay=0.01, betas=(0.9, 0.98)
    )

    for _ in (pbar := tqdm.trange(epochs)):
        epoch_loss = 0
        for _, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            optim.zero_grad()
            out = model(input_ids, attention_mask=attention_mask)
            N, d, C = out.shape
            labels = batch["labels"].T.to(device)
            loss = criterion(out.view(N, C, d), labels)
            epoch_loss += loss.item()

            loss.backward()
            optim.step()

        val_loss = evaluate(model, val_loader)
        pbar.set_postfix(
            {"val_loss": val_loss, "train_loss": epoch_loss / len(train_loader)}
        )


train(model, train_loader, val_loader)
