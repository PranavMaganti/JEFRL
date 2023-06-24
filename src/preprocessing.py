import argparse
import json
import logging
import os
from pathlib import Path
import pickle
import sys

from js_ast.analysis import scope_analysis
from preprocessing.filter import filter_corpus_by_coverage
from preprocessing.filter import filter_corpus_by_length
from preprocessing.normalise import normalize_ast
from preprocessing.sanitise import sanitise_ast
from preprocessing.vocabulary import get_frag_counts
from preprocessing.vocabulary import get_frag_data
from preprocessing.vocabulary import get_vocab
from tqdm import tqdm
from transformer.tokenizer import ASTTokenizer

from utils.js_engine import Engine
from utils.js_engine import V8Engine
from utils.loader import get_subtrees
from utils.loader import load_corpus
from utils.logging import setup_logging


parser = argparse.ArgumentParser(
    prog="JEFRL Preprocessing", description="Preprocesses the corpus for JEFRL"
)

parser.add_argument(
    "--engine-name", type=str, default="v8", help="The engine to use for preprocessing"
)
parser.add_argument(
    "--engine-executable",
    type=Path,
    default="engines/v8/v8/out/fuzzbuild/d8",
    help="The path to the engine to use for preprocessing",
)
parser.add_argument(
    "--corpus",
    type=str,
    default=Path("corpus/DIE"),
    help="The path to the corpus to preprocess",
)
parser.add_argument(
    "--corpus-config",
    type=str,
    default=Path("corpus/configs/DIE.json"),
    help="The path to the corpus config file",
)
parser.add_argument(
    "--output",
    type=str,
    default=Path("data/"),
    help="The path to the output directory where processed files will be stored",
)
args = parser.parse_args()

# System setup
sys.setrecursionlimit(10000)

# Logging setup
setup_logging()


if not args.engine_executable.exists():
    raise FileNotFoundError(f"Engine not found: {args.engine_executable}")

if not args.corpus.exists():
    raise FileNotFoundError(f"Corpus not found: {args.corpus}")

if not args.corpus_config.exists():
    raise FileNotFoundError(f"Corpus config not found: {args.corpus_config}")

if not args.output.exists():
    os.makedirs(args.output, exist_ok=True)

with open(args.corpus_config, "r") as f:
    corpus_config = json.load(f)

engine = Engine.get_engine(args.engine_name, args.engine_executable)

V8Engine(args.engine_executable)
corpus = load_corpus(engine, args.corpus, corpus_config)

for state in tqdm(corpus, desc="Normalising corpus"):
    sanitise_ast(state.root)
    normalize_ast(state.root)

# Get subtrees before length filtering as we can still use subtrees from long programs
subtrees = get_subtrees(corpus)
corpus = filter_corpus_by_length(corpus)


# Get data for pre-training AST Transformer
frag_seqs, node_types = get_frag_data(corpus)
frag_counts = get_frag_counts(frag_seqs)
vocab, token_to_id, id_to_token, special_token_ids = get_vocab(frag_counts, node_types)

tokenizer = ASTTokenizer(vocab, token_to_id)
frag_data = [tokenizer.frag_seq_to_ids(frag_seq) for frag_seq in frag_seqs]

corpus, total_coverage = filter_corpus_by_coverage(corpus)

for state in tqdm(corpus, desc="Analysing corpus"):
    scope_analysis(state.root)

logging.info(f"Total coverage: {total_coverage}")

with open(args.output / "corpus.pkl", "wb") as f:
    pickle.dump(
        {"corpus": corpus, "subtrees": subtrees, "total_coverage": total_coverage}, f
    )


with open(args.output / "vocab_data.pkl", "wb") as f:
    pickle.dump(
        {
            "vocab": vocab,
            "token_to_id": token_to_id,
            "id_to_token": id_to_token,
            "special_token_ids": special_token_ids,
        },
        f,
    )


with open(args.output / "frag_data.pkl", "wb") as f:
    pickle.dump({"frag_data": frag_data, "node_types": node_types}, f)
