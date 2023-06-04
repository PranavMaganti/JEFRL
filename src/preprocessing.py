import pickle
import sys

from js_ast.analysis import scope_analysis
from preprocessing.filter import filter_corpus_by_coverage
from preprocessing.normalise import collect_id
from preprocessing.normalise import normalize_id
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
corpus = load_corpus(engine)

for state in tqdm.tqdm(corpus, desc="Normalising corpus"):
    id_dict: dict[str, str] = {}
    collect_id(state.program, id_dict, {"v": 0, "f": 0, "c": 0})
    normalize_id(state.program, id_dict)

subtrees = get_subtrees(corpus)
corpus, total_coverage = filter_corpus_by_coverage(corpus)

print(f"Total coverage: {total_coverage}")

for state in tqdm.tqdm(corpus, desc="Analysing corpus"):
    scope_analysis(state.program)


with open("data/js-rl/corpus.pkl", "wb") as f:
    pickle.dump(
        {"corpus": corpus, "subtrees": subtrees, "total_coverage": total_coverage}, f
    )
