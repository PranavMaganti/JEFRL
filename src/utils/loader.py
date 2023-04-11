import copy
import glob
import logging
from collections import defaultdict

import esprima
import tqdm

from js_ast.nodes import Node


def load_corpus(path: str) -> list[Node]:
    files = glob.glob(f"{path}/*.js")
    logging.info(f"Found {len(files)} files in corpus")
    corpus = []

    for file in tqdm.tqdm(files):
        with open(file, "r") as f:
            code = f.read()
        try:
            ast = esprima.parseScript(code, tolerant=True, jsx=True)
            ast = Node.from_dict(ast.toDict())
        except esprima.error_handler.Error:  # type: ignore
            # logging.warning(f"Failed to parse {file}")
            continue

        corpus.append(ast)

    logging.info(f"Loaded {len(corpus)}/{len(files)} files from corpus")

    return corpus


def get_subtrees(corpus: list[Node]) -> dict[str, list[Node]]:
    subtrees = defaultdict(list)

    for ast in corpus:
        for node in ast.traverse():
            if hasattr(node, "type"):
                subtrees[node.type].append(copy.deepcopy(node))

    return subtrees
