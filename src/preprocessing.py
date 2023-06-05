import dataclasses
from dataclasses import dataclass
from pathlib import Path
import pickle
import sys

from js_ast.analysis import scope_analysis
from js_ast.nodes import ClassDeclaration
from js_ast.nodes import Expression
from js_ast.nodes import FunctionDeclaration
from js_ast.nodes import Node
from preprocessing.filter import filter_corpus_by_coverage
from preprocessing.filter import filter_corpus_by_length
from preprocessing.normalise import collect_id
from preprocessing.normalise import normalize_ast
from preprocessing.normalise import normalize_id
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

engine = V8Engine()
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


with open("data/js-rl/corpus.pkl", "wb") as f:
    pickle.dump(
        {"corpus": corpus, "subtrees": subtrees, "total_coverage": total_coverage}, f
    )
