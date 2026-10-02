from pathlib import Path
from typing import Any, Optional

from rl.env import MAX_FRAGMENT_SEQ_LEN
import torch
from transformer.special_tokens import CLS_TOKEN
from transformer.special_tokens import PAD_TOKEN
from transformer.special_tokens import SEP_TOKEN
from transformer.tokenizer import ASTTokenizer
from transformers import RobertaConfig
from transformers import RobertaModel


intermediate_size = 2048  # embedding dimension
hidden_size = 512

num_hidden_layers = 3
num_attention_heads = 8
dropout = 0


def get_ast_transformer_config(
    vocab_size: int, token_to_id: dict[str, int], dropout: float = dropout
) -> RobertaConfig:
    return RobertaConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        intermediate_size=intermediate_size,
        hidden_dropout_prob=dropout,
        # Previously left at RoBERTa's default of 0.1 despite the tuned dropout of 0
        # (it is also unsupported by MPS's fused attention when training)
        attention_probs_dropout_prob=dropout,
        # RoBERTa defaults to pad=1, bos=0, eos=2, but our vocabulary puts <pad> at
        # 0 and <s> (CLS) at 1. With the defaults the CLS embedding is treated as
        # padding: it is fixed at zero, never trained, and positions are off by one.
        pad_token_id=token_to_id[PAD_TOKEN],
        bos_token_id=token_to_id[CLS_TOKEN],
        eos_token_id=token_to_id[SEP_TOKEN],
        # Positions start after the padding index
        max_position_embeddings=MAX_FRAGMENT_SEQ_LEN + token_to_id[PAD_TOKEN] + 1,
    )


def _extract_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    """Accepts a saved state dict or a whole saved model (the old pretraining script
    saved whole RobertaForMaskedLM objects)."""
    if isinstance(checkpoint, torch.nn.Module):
        checkpoint = checkpoint.state_dict()

    state_dict: dict[str, torch.Tensor] = {}
    for key, value in checkpoint.items():
        if key.startswith("lm_head."):
            continue
        state_dict[key.removeprefix("roberta.")] = value
    return state_dict


def get_ast_transformer_model(
    vocab: set[str],
    token_to_id: dict[str, int],
    pretrained_model_path: Optional[Path],
    device: torch.device,
) -> tuple[ASTTokenizer, RobertaModel, RobertaConfig]:
    config = get_ast_transformer_config(len(vocab), token_to_id)
    tokenizer = ASTTokenizer(vocab, token_to_id, MAX_FRAGMENT_SEQ_LEN)

    # The pooler is not trained by masked language modelling, so it is left out and
    # the embedding is a mean over the final hidden states instead (see embed)
    ast_net = RobertaModel(config, add_pooling_layer=False)

    if pretrained_model_path is not None:
        checkpoint = torch.load(pretrained_model_path, map_location="cpu", weights_only=False)
        state_dict = _extract_state_dict(checkpoint)
        # Checkpoints from before the pooler was dropped still contain it
        state_dict = {k: v for k, v in state_dict.items() if not k.startswith("pooler.")}
        missing, unexpected = ast_net.load_state_dict(state_dict, strict=False)
        # position_ids is a buffer some transformers versions don't save
        missing = [k for k in missing if not k.endswith("position_ids")]
        if missing or unexpected:
            raise ValueError(
                f"Checkpoint {pretrained_model_path} does not match the AST Transformer: "
                f"missing {missing}, unexpected {unexpected}"
            )

    return tokenizer, ast_net.to(device), config


def embed(
    ast_net: RobertaModel, input_ids: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    """Mean of the final hidden states over the non-padding tokens."""
    hidden = ast_net(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
    mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
