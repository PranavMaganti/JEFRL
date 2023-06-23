import argparse
import dataclasses
from dataclasses import dataclass
import os
from pathlib import Path
import pickle
import sys

from js_ast.analysis import scope_analysis
from preprocessing.filter import filter_corpus_by_coverage
from preprocessing.filter import filter_corpus_by_length
from preprocessing.normalise import normalize_ast
from preprocessing.sanitise import sanitise_ast
import tqdm

from utils.js_engine import V8Engine
from utils.loader import get_subtrees
from utils.loader import load_corpus
from utils.logging import setup_logging


# System setup
sys.setrecursionlimit(10000)

# Logging setup
setup_logging()

os.makedirs("data/", exist_ok=True)

parser = argparse.ArgumentParser(
    prog="JEFRL Preprocessing", description="Preprocesses the corpus for JEFRL"
)

# parser.add_argument('--engine', type=str, default='v8', help='The engine to use for preprocessing')
# parser.add_argument('--engine-path', type=str, default="engines/v8/v8/out/fuzzbuild/d8", help='The path to the engine to use for preprocessing')
parser.add_argument(
    "--version",
    type=str,
    default="latest",
    help="The version of the engine to use for preprocessing",
)
# parser.add_argument('--corpus-path', type=str, default='corpus/DIE', help='The path to the corpus to preprocess')
# parser.add_argument('--output-path', type=str, default='data/', help='The path to the output directory where processed files will be stored')
# parser.add_argument('--corpus-config', type=str, default='corpus/config', help='The path to the corpus config file')

args = parser.parse_args()

engine = V8Engine(version=args.version)
corpus = load_corpus(engine, Path("corpus/DIE"))

for state in tqdm.tqdm(corpus, desc="Normalising corpus"):
    sanitise_ast(state.program)
    normalize_ast(state.program)


subtrees = get_subtrees(corpus)
corpus = filter_corpus_by_length(corpus)
corpus, total_coverage = filter_corpus_by_coverage(corpus)

print(f"Total coverage: {total_coverage}")

for state in tqdm.tqdm(corpus, desc="Analysing corpus"):
    scope_analysis(state.program)


with open(f"data/corpus.pkl", "wb") as f:
    pickle.dump(
        {"corpus": corpus, "subtrees": subtrees, "total_coverage": total_coverage}, f
    )
