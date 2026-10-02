from pathlib import Path

import torch
from transformer.ast_transformer import embed
from transformer.ast_transformer import get_ast_transformer_config
from transformer.ast_transformer import get_ast_transformer_model
from transformer.special_tokens import CLS_TOKEN
from transformer.special_tokens import PAD_TOKEN
from transformer.special_tokens import special_tokens
from transformers import RobertaForMaskedLM


VOCAB = special_tokens + [f"frag{i}" for i in range(20)]
TOKEN_TO_ID = {t: i for i, t in enumerate(VOCAB)}
CPU = torch.device("cpu")


def test_cls_token_is_not_treated_as_padding():
    config = get_ast_transformer_config(len(VOCAB), TOKEN_TO_ID)
    assert config.pad_token_id == TOKEN_TO_ID[PAD_TOKEN] == 0
    assert config.bos_token_id == TOKEN_TO_ID[CLS_TOKEN] == 1

    model = RobertaForMaskedLM(config)
    embeddings = model.roberta.embeddings.word_embeddings
    # nn.Embedding keeps the padding row at zero and never trains it
    assert embeddings.padding_idx == TOKEN_TO_ID[PAD_TOKEN]
    assert embeddings.weight[TOKEN_TO_ID[CLS_TOKEN]].abs().sum() > 0


def test_max_length_sequence_fits_position_embeddings():
    config = get_ast_transformer_config(len(VOCAB), TOKEN_TO_ID)
    model = RobertaForMaskedLM(config)
    ids = torch.full((1, 512), 7)
    ids[0, 0] = TOKEN_TO_ID[CLS_TOKEN]
    # Raises an index error if positions overflow
    model(input_ids=ids, attention_mask=torch.ones_like(ids))


def test_loads_old_whole_model_and_new_state_dict_checkpoints(tmp_path: Path):
    config = get_ast_transformer_config(len(VOCAB), TOKEN_TO_ID)
    mlm = RobertaForMaskedLM(config)

    torch.save(mlm.state_dict(), tmp_path / "state_dict.pt")
    torch.save(mlm, tmp_path / "whole_model.pt")

    for name in ("state_dict.pt", "whole_model.pt"):
        _, ast_net, _ = get_ast_transformer_model(
            set(VOCAB), TOKEN_TO_ID, tmp_path / name, CPU
        )
        assert torch.equal(
            ast_net.embeddings.word_embeddings.weight,
            mlm.roberta.embeddings.word_embeddings.weight,
        )


def test_embedding_ignores_padding():
    _, ast_net, config = get_ast_transformer_model(set(VOCAB), TOKEN_TO_ID, None, CPU)
    ast_net.eval()
    seq = torch.tensor([[1, 7, 8, 9, 3]])
    padded = torch.tensor([[1, 7, 8, 9, 3, 0, 0, 0]])

    with torch.no_grad():
        a = embed(ast_net, seq, torch.ones_like(seq))
        b = embed(ast_net, padded, (padded != 0).long())

    assert a.shape == (1, config.hidden_size)
    assert torch.allclose(a, b, atol=1e-5)
