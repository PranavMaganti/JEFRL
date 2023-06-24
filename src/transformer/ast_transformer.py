from pathlib import Path

from rl.env import MAX_FRAGMENT_SEQ_LEN
import torch
from transformer.tokenizer import ASTTokenizer
from transformers import RobertaConfig
from transformers import RobertaModel


intermediate_size = 2048  # embedding dimension
hidden_size = 512

num_hidden_layers = 3
num_attention_heads = 8
dropout = 0


def get_ast_transformer_config(vocab_size: int) -> RobertaConfig:
    return RobertaConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        intermediate_size=intermediate_size,
        hidden_dropout_prob=dropout,
        max_position_embeddings=MAX_FRAGMENT_SEQ_LEN + 2,
    )


def get_ast_transformer_model(
    vocab: set[str],
    token_to_id: dict[str, int],
    pretrained_model_path: Path,
    device: torch.device,
) -> tuple[ASTTokenizer, RobertaModel, RobertaConfig]:
    config = RobertaConfig(
        vocab_size=len(vocab),
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        intermediate_size=intermediate_size,
        hidden_dropout_prob=dropout,
        max_position_embeddings=MAX_FRAGMENT_SEQ_LEN + 2,
    )

    # Load the ASTBERTa model
    tokenizer = ASTTokenizer(vocab, token_to_id, MAX_FRAGMENT_SEQ_LEN)

    pretrained_model = torch.load(pretrained_model_path)
    ast_net = RobertaModel.from_pretrained(
        pretrained_model_name_or_path=None,
        state_dict=pretrained_model.state_dict(),
        config=config,
    ).to(device)

    assert isinstance(config, RobertaConfig)
    assert isinstance(ast_net, RobertaModel)

    return tokenizer, ast_net, config
