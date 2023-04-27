import copy
import glob
import logging
import pickle
from collections import defaultdict
from pathlib import Path

import esprima
import tqdm

from js_ast.nodes import Node
from rl.env import ProgramState
from utils.js_engine import Engine, JSError


def load_corpus(engine: Engine) -> list[ProgramState]:
    path = engine.get_corpus()
    files = glob.glob(f"{path}/*.js")
    logging.info(f"Found {len(files)} files in corpus")
    corpus = []

    for file in tqdm.tqdm(files):
        exec_data_path = Path(file).with_suffix(".pkl")

        with open(file, "r") as f:
            code = f.read()

        if not exec_data_path.exists():
            exec_data = engine.execute_text(code)
            with open(exec_data_path, "wb") as f:
                pickle.dump(exec_data, f)
        else:
            with open(exec_data_path, "rb") as f:
                exec_data = pickle.load(f)

        if exec_data is None or exec_data.error != JSError.NoError:
            logging.info(f"Failed to execute {file} or produced an error")
            continue

        try:
            ast = esprima.parseScript(code, tolerant=True, jsx=True)
            ast = Node.from_dict(ast.toDict())
        except esprima.error_handler.Error:  # type: ignore
            # logging.warning(f"Failed to parse {file}")
            continue

        corpus.append(ProgramState(ast, exec_data.coverage_data))  # type: ignore

    logging.info(f"Loaded {len(corpus)}/{len(files)} files from corpus")
    return corpus


def get_subtrees(corpus: list[ProgramState]) -> dict[str, list[Node]]:
    subtrees = defaultdict(list)

    for state in corpus:
        for node in state.current_node.traverse():
            if hasattr(node, "type"):
                subtrees[node.type].append(copy.deepcopy(node))

    return subtrees
