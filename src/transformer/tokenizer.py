from datetime import datetime
from typing import Any, Iterable, Optional

from js_ast.fragmentise import hash_frag
from js_ast.fragmentise import node_to_frags
from js_ast.nodes import Node
import torch
from transformer.special_tokens import CLS_TOKEN
from transformer.special_tokens import PAD_TOKEN
from transformer.special_tokens import SEP_TOKEN
from transformer.special_tokens import UNK_TOKEN


class ASTTokenizer:
    def __init__(
        self,
        vocab: set[str],
        token_to_id: dict[str, int],
        max_len: Optional[int] = None,
    ):
        self.vocab = vocab
        self.token_to_id = token_to_id
        self.max_len = max_len

    def tokenize(self, ast: Node) -> list[int]:
        frag_seq: list[dict[str, Any]] = []
        node_types: set[str] = set()
        node_to_frags(ast, frag_seq, node_types, max_len=self.max_len)

        return self.frag_seq_to_ids(frag_seq)

    def frag_seq_to_ids(self, frag_seq: list[dict[str, Any]]) -> list[int]:
        frag_id_seq: list[int] = []
        frag_id_seq.append(self.token_to_id[CLS_TOKEN])

        for frag in frag_seq:
            frag_hash = hash_frag(frag)
            if frag_hash in self.token_to_id:
                frag_id_seq.append(self.token_to_id[frag_hash])
            else:
                oov_frag: dict[str, str] = {"type": frag["type"]}
                oov_frag_hash = hash_frag(oov_frag)
                if oov_frag_hash in self.token_to_id:
                    frag_id_seq.append(self.token_to_id[oov_frag_hash])
                else:
                    print(f"UNK_TOKEN: {frag_hash}")
                    frag_id_seq.append(self.token_to_id[UNK_TOKEN])

        if self.max_len is None or len(frag_id_seq) < self.max_len:
            frag_id_seq.append(self.token_to_id[SEP_TOKEN])

        return frag_id_seq

    def pad_batch(
        self, batch: list[torch.Tensor], device: torch.device
    ) -> dict[str, torch.Tensor]:
        inputs = torch.nn.utils.rnn.pad_sequence(list(batch), batch_first=True).to(
            device
        )
        attention_mask = torch.ones_like(inputs, device=device)
        attention_mask[inputs == self.token_to_id[PAD_TOKEN]] = 0

        return {
            "input_ids": inputs,
            "attention_mask": attention_mask,
        }
