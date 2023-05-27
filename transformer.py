import pickle

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


def seq_data_collator(
    batch: list[tuple[list[int], list[str]]]
) -> dict[str, torch.Tensor]:
    seqs, frag_types = zip(*batch)

    max_len = min(max(map(lambda x: len(x), batch)), MAX_SEQ_LEN)

    padded_seqs: list[list[int]] = []
    padded_frag_types: list[list[str]] = []

    for seq, seq_types in zip(seqs, frag_types):
        if len(seq) > max_len:
            # start_idx = random.randint(0, len(seq))
            # length = random.randint(1, max_len)
            # end_idx = min(start_idx + length, len(seq))
            seq = seq[:max_len]
            seq_types = seq_types[:max_len]

        padded_seq = seq + [token_to_id[PAD_TOKEN]] * (max_len - len(seq))
        padded_seq_types = seq_types + [PAD_TOKEN] * (max_len - len(seq))

        padded_seqs.append(padded_seq)
        padded_frag_types.append(padded_seq_types)

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
        "frag_types": padded_frag_types,
    }


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vocab_size = len(vocab)  # size of vocabulary
intermediate_size = 1200  # embedding dimension
hidden_size = (
    400  # dimension of the feedforward network model in ``nn.TransformerEncoder``
)

num_hidden_layers = (
    12  # number of ``nn.TransformerEncoderLayer`` in ``nn.TransformerEncoder``
)
num_attention_heads = 12  # number of heads in ``nn.MultiheadAttention``
dropout = 0.1  # dropout probability

batch_size = 32

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

print(
    f"The model has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters"
)
print(model)


import tqdm
from torch import nn


def evaluate_batch(model: RobertaForMaskedLM, batch: dict[str, Any]) -> torch.Tensor:
    criterion = nn.CrossEntropyLoss()

    input_ids = batch["input_ids"].to(device)
    labels = batch["labels"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    batch_types = batch["frag_types"]

    out = model(input_ids, attention_mask=attention_mask).logits
    out = torch.nn.functional.softmax(out, dim=-1)
    N, d, C = out.shape

    l1 = criterion(out.view(N, C, d), labels)

    # top_k_sum: torch.Tensor = torch.topk(out, top_k, dim=-1).values.sum(dim=-1)  # type: ignore
    # type_sum = torch.zeros_like(top_k_sum)

    # for i, seq_types in enumerate(batch_types):
    #     for j, frag_type in enumerate(seq_types):
    #         target_idx = frag_type_to_ids[frag_type]
    #         type_sum[i, j] = out[i, j, target_idx].sum()

    # total_l2 = top_k_sum - type_sum

    # labels_mask = batch["labels"] == -100
    # total_l2[labels_mask] = 0

    # l2 = total_l2.mean()
    # loss = l1 + l2
    loss = l1

    return loss


def evaluate(model: RobertaForMaskedLM, val_loader: DataLoader[list[int]]):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            loss = evaluate_batch(model, batch)
            total_loss += loss.item()

    return total_loss / len(val_loader)


def train(
    model: RobertaForMaskedLM,
    train_loader: DataLoader[list[int]],
    val_loader: DataLoader[list[int]],
):
    model.train()
    epochs = 20
    optim = torch.optim.AdamW(
        model.parameters(), lr=6e-4, eps=1e-6, weight_decay=0.01, betas=(0.9, 0.98)
    )

    for _ in (pbar := tqdm.trange(epochs)):
        epoch_loss = 0
        for _, batch in (
            ibar := tqdm.tqdm(
                enumerate(train_loader), leave=False, total=len(train_loader)
            )
        ):
            loss = evaluate_batch(model, batch)
            epoch_loss += loss.item()

            ibar.set_postfix({"loss": loss.item()})

            loss.backward()
            optim.step()

        val_loss = evaluate(model, val_loader)
        pbar.set_postfix(
            {"val_loss": val_loss, "train_loss": epoch_loss / len(train_loader)}
        )


train(model, train_loader, val_loader)
