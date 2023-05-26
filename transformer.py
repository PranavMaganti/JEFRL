import math
import pickle

import torch
from torch import Tensor, nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import DataLoader, Dataset, random_split

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


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe: torch.Tensor = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


class TransformerModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        intermediate_size: int,
        num_attention_heads: int,
        hidden_size: int,
        num_layers: int,
        dropout: float = 0.1,
        initializer_range: float = 0.02,
    ):
        super().__init__()
        self.model_type = "Transformer"
        self.pos_encoder = PositionalEncoding(intermediate_size, dropout)
        encoder_layers = TransformerEncoderLayer(
            intermediate_size, num_attention_heads, hidden_size, dropout
        )
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)
        self.encoder = nn.Embedding(vocab_size, intermediate_size)
        self.intermediate_size = intermediate_size
        self.decoder = nn.Linear(intermediate_size, vocab_size)

        self.initializer_range = initializer_range

        self.init_weights()

    def init_weights(self) -> None:
        self.encoder.weight.data.uniform_(
            -self.initializer_range, self.initializer_range
        )
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(
            -self.initializer_range, self.initializer_range
        )

    def forward(
        self, src: Tensor, src_mask: Tensor, src_key_padding_mask: Tensor
    ) -> Tensor:
        """
        Arguments:
            src: Tensor, shape ``[seq_len, batch_size]``
            src_mask: Tensor, shape ``[seq_len, seq_len]``

        Returns:
            output Tensor of shape ``[seq_len, batch_size, ntoken]``
        """
        src = self.encoder(src) * math.sqrt(self.intermediate_size)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask, src_key_padding_mask)
        output = self.decoder(output)
        return output

    def generate_square_subsequent_mask(self, sz: int) -> Tensor:
        """Generates an upper-triangular matrix of ``-inf``, with zeros on ``diag``."""
        return torch.triu(torch.ones(sz, sz) * float("-inf"), diagonal=1)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

vocab_size = len(vocab)  # size of vocabulary
intermediate_size = 3072  # embedding dimension
hidden_size = 768  # dimension of the feedforward network model in ``nn.TransformerEncoder``
num_layers = 12  # number of ``nn.TransformerEncoderLayer`` in ``nn.TransformerEncoder``
num_attention_heads = 12  # number of heads in ``nn.MultiheadAttention``
dropout = 0.1  # dropout probability

batch_size = 32

dataset = FragDataset(data)
train_split, val_split, test_split = random_split(dataset, [0.8, 0.1, 0.1])

train_loader = DataLoader(train_split, batch_size=batch_size, shuffle=True, collate_fn=seq_data_collator)
val_loader = DataLoader(val_split, batch_size=batch_size, shuffle=True, collate_fn=seq_data_collator)
test_loader = DataLoader(test_split, batch_size=batch_size, shuffle=True, collate_fn=seq_data_collator)

model = TransformerModel(vocab_size, intermediate_size, num_attention_heads, hidden_size, num_layers, dropout).to(device)
print(f'The model has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters')

def evaluate(model: TransformerModel, val_loader: DataLoader[list[int]]):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            seq_len = batch["input_ids"].shape[1]
            src_mask = model.generate_square_subsequent_mask(seq_len).to(device)
            input_ids = batch["input_ids"].T.to(device)

            out = model(input_ids, src_mask, batch["attention_mask"].to(device))
            N, d, C = out.shape
            labels = batch['labels'].T.to(device)
            loss = criterion(out.view(N, C, d), labels)
            total_loss += loss.item()
    
    return total_loss / len(val_loader)
            

def train(model: TransformerModel, train_loader: DataLoader[list[int]], val_loader: DataLoader[list[int]]):
    model.train()
    epochs = 500
    criterion = nn.CrossEntropyLoss()
    optim = torch.optim.AdamW(model.parameters(), lr=6e-4, eps=1e-6, weight_decay=0.01, betas=(0.9, 0.98))

    for _ in (pbar := tqdm.trange(epochs)):
        epoch_loss = 0
        for _, batch in enumerate(train_loader):
            seq_len = batch["input_ids"].shape[1]
            src_mask = model.generate_square_subsequent_mask(seq_len).to(device)
            input_ids = batch["input_ids"].T.to(device)

            optim.zero_grad()
            out = model(input_ids, src_mask, batch["attention_mask"].to(device))
            N, d, C = out.shape
            labels = batch['labels'].T.to(device)
            loss = criterion(out.view(N, C, d), labels)
            epoch_loss += loss.item()

            loss.backward()
            optim.step()


        val_loss = evaluate(model, val_loader)
        pbar.set_postfix({"val_loss": val_loss, "train_loss": epoch_loss / len(train_loader)})

train(model, train_loader, val_loader)