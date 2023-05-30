from pathlib import Path
import pickle
from typing import Any

import torch
import tqdm
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import RobertaConfig, RobertaForMaskedLM

with open("ASTBERTa/vocab.pkl", "rb") as f:
    vocab = pickle.load(f)

with open("ASTBERTa/token_to_id.pkl", "rb") as f:
    token_to_id = pickle.load(f)

with open("ASTBERTa/data.pkl", "rb") as f:
    data = pickle.load(f)

with open("ASTBERTa/frag_type_to_ids.pkl", "rb") as f:
    frag_type_to_ids = pickle.load(f)


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
MODEL_SAVE_PATH = "ASTBERTa/models-l1/"


def seq_data_collator(
    batch: list[tuple[list[int], list[list[int]]]]
) -> dict[str, torch.Tensor]:
    seqs, _ = zip(*batch)
    # max_len = min(max(map(lambda x: len(x), seqs)), MAX_SEQ_LEN)

    # processed_seqs: list[list[int]] = []
    # processed_frag_idxs: list[torch.Tensor] = []
    # processed_frag_idxs_mask: list[torch.Tensor] = []

    # for seq, frag_idxs in zip(seqs, possible_frag_idxs):
    #   if len(seq) > max_len:
    #     # start_idx = random.randint(0, len(seq))
    #     # length = random.randint(1, max_len)
    #     # end_idx = min(start_idx + length, len(seq))
    #     seq = seq[:max_len]
    #     frag_idxs = frag_idxs[:max_len]

    # padded_seq = seq + [token_to_id[PAD_TOKEN]] * (max_len - len(seq))
    # padded_frag_idxs = frag_idxs + [[]  for _ in range(max_len - len(seq))]
    # padded_frag_idxs = torch.nested.nested_tensor([torch.tensor(l, dtype=torch.int64) for l in padded_frag_idxs]).to_padded_tensor(-100)

    # frag_idxs_mask = padded_frag_idxs == -100
    # padded_frag_idxs.masked_fill_(frag_idxs_mask, value=0)

    # processed_seqs.append(padded_seq)
    # processed_frag_idxs.append(padded_frag_idxs)
    # processed_frag_idxs_mask.append(frag_idxs_mask)

    inputs = torch.nn.utils.rnn.pad_sequence(
        [torch.tensor(x[:MAX_SEQ_LEN]) for x in seqs], batch_first=True
    )
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
        # "possible_frag_idxs": processed_frag_idxs,
        # "possible_frag_idxs_mask": processed_frag_idxs_mask
    }


def evaluate_batch(model: RobertaForMaskedLM, batch: dict[str, Any]) -> torch.Tensor:
    criterion = nn.CrossEntropyLoss()

    input_ids = batch["input_ids"].to(device)
    labels = batch["labels"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    # possible_frag_idxs = batch["possible_frag_idxs"]
    # possible_frag_idxs_mask = batch["possible_frag_idxs_mask"]

    out = model(input_ids, attention_mask=attention_mask).logits
    l1 = criterion(out.view(-1, vocab_size), labels.view(-1))

    # preds = nn.functional.softmax(out, dim=-1)
    # top_k_preds: torch.Tensor = torch.topk(preds, top_k, dim=-1)  # type: ignore
    # top_k_sum = top_k_preds.values.sum(dim=-1).view(-1)

    # type_sum = torch.tensor([], device=device)

    # for i, seq_preds in enumerate(preds):
    #     type_preds = torch.gather(seq_preds, 1, possible_frag_idxs[i].to(device))
    #     type_preds[possible_frag_idxs_mask[i].to(device)] = 0.0
    #     type_preds = type_preds.sum(dim=1)
    #     type_sum = torch.cat((type_sum, type_preds))

    # total_l2 = top_k_sum - type_sum
    # labels_mask = (batch["labels"] != -100).to(device).view(-1).float()
    # l2 = (total_l2 * labels_mask).sum() / labels_mask.sum()

    # loss = (l1 + l2).to(device)
    # loss = l1

    return l1.to(device)


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
    model_save_path: Path = Path(MODEL_SAVE_PATH),
    epochs: int = 8,
):
    steps = 0
    epochs = 20

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
            optim.zero_grad()
            loss = evaluate_batch(model, batch)
            loss.backward()
            optim.step()
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

batch_size = 14
top_k = 64

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
# model = torch.nn.DataParallel(model, device_ids=[0])

optim = torch.optim.AdamW(
    model.parameters(),
    lr=1e-3,
    eps=1e-6,
    weight_decay=0.01,
    betas=(0.9, 0.98),
)

print(
    f"The model has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters"
)
print(model)

train(model, train_loader, val_loader, optim)
