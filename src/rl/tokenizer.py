from typing import Any, Iterable

import torch
from js_ast.fragmentise import hash_frag, node_to_frags
from js_ast.nodes import Node

PAD_TOKEN = "<pad>"
CLS_TOKEN = "<s>"
SEP_TOKEN = "</s>"
MASK_TOKEN = "<mask>"
UNK_TOKEN = "<unk>"


class ASTTokenizer:
    def __init__(self, vocab: set[str], token_to_id: dict[str, int], max_len: int):
        self.vocab = vocab
        self.token_to_id = token_to_id
        self.max_len = max_len

    def tokenize(self, ast: Node) -> torch.Tensor:
        frag_seq: list[dict[str, Any]] = []
        frag_info_seq = []
        node_types: set[str] = set()

        node_to_frags(ast, frag_seq, frag_info_seq, node_types)

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

            if len(frag_id_seq) >= self.max_len:
                break

        if len(frag_id_seq) < self.max_len:
            frag_id_seq.append(self.token_to_id[SEP_TOKEN])

        return torch.tensor(frag_id_seq, dtype=torch.long)

    def process_batch(
        self, batch: Iterable[torch.Tensor], device: torch.device
    ) -> dict[str, torch.Tensor]:
        inputs = torch.nn.utils.rnn.pad_sequence(list(batch), batch_first=True)
        attention_mask = torch.ones_like(inputs)
        attention_mask[inputs == self.token_to_id[PAD_TOKEN]] = 0

        return {
            "input_ids": inputs.to(device),
            "attention_mask": attention_mask.to(device),
        }
