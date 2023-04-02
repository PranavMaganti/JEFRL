from collections import defaultdict

from environment.main import FuzzingEnv, ProgramState
from utils.js_engine import ExecutionData
from utils.loader import load_corpus

CORPUS_PATH = "corpus"

corpus = load_corpus(CORPUS_PATH)
subtrees = defaultdict(list)
for i, ast in enumerate(corpus):
    for node in ast.traverse():
        if hasattr(node, "type"):
            subtrees[node.type].append(node)

seeds = []

for ast in corpus:
    execution_data = ExecutionData(0, 0, 0)
    seeds.append(ProgramState(ast, execution_data))


env = FuzzingEnv(5, 5, seeds, subtrees)
