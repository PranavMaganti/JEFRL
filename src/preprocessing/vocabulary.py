from collections import defaultdict
from platform import node
from typing import Any

from js_ast.fragmentise import hash_frag
from js_ast.fragmentise import node_to_frags
from rl.program_state import ProgramState
from traitlets import default
from transformer.special_tokens import CLS_TOKEN
from transformer.special_tokens import PAD_TOKEN
from transformer.special_tokens import SEP_TOKEN
from transformer.special_tokens import special_tokens
from transformer.special_tokens import UNK_TOKEN


def get_frag_data(corpus: list[ProgramState]):
    frag_seqs: list[list[dict[str, Any]]] = []
    all_node_types: set[str] = set()

    for state in corpus:
        frag_seq: list[dict[str, Any]] = []
        node_types: set[str] = set()

        node_to_frags(state.root, frag_seq, node_types)
        frag_seqs.append(frag_seq)
        all_node_types |= node_types

    return frag_seqs, all_node_types


def get_frag_counts(frag_seqs: list[list[dict[str, Any]]]):
    frag_counts: dict[str, int] = defaultdict(int)

    for frag_seq in frag_seqs:
        for frag in frag_seq:
            frag_str = hash_frag(frag)
            frag_counts[frag_str] += 1

    return frag_counts


def get_vocab(
    frag_counts: dict[str, int],
    node_types: set[str],
    min_count: int = 3,
):
    frag_freq_list = list(sorted(frag_counts.items(), reverse=True, key=lambda x: x[1]))
    oov_frags: list[str] = []

    # Add OOV anonymous frag type for those not in vocabulary
    for frag_type in node_types:
        oov_frag = {"type": frag_type}
        oov_frag_hash = hash_frag(oov_frag)
        oov_frags.append(oov_frag_hash)

    threshold_frags = [
        frag_hash for frag_hash, freq in frag_freq_list if freq >= min_count
    ]
    threshold_frags = threshold_frags[: 20000 - len(oov_frags)]

    vocab_frags = set(threshold_frags + oov_frags)

    ordered_vocab = special_tokens + list(vocab_frags)
    vocab = set(ordered_vocab)

    print(len(vocab))

    token_to_id = {token: i for i, token in enumerate(ordered_vocab)}
    id_to_token = {i: token for token, i in token_to_id.items()}

    special_token_ids = set([token_to_id[token] for token in special_tokens])

    return vocab, token_to_id, id_to_token, special_token_ids
