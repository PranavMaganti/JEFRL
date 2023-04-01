from collections import defaultdict
import json
from pprint import pprint
from re import sub
from nodes.main import objectify
from nodes.main import Node

from utils.loader import load_corpus

CORPUS_PATH = "corpus"

corpus = load_corpus(CORPUS_PATH)
trees: list[Node] = [objectify(ast.toDict()) for ast in corpus]  # type: ignore
subtrees = defaultdict(list)
for i, ast in enumerate(trees):
    for node in ast.traverse():
        if hasattr(node, "type"):
            subtrees[node.type].append(node)

