import glob
import logging
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any

import esprima
import tqdm

from js_ast.nodes import (CallExpression, ExpressionStatement, Node,
                          UnknownNodeTypeError)
from rl.env import ProgramState
from utils.js_engine import Engine, JSError


def load_corpus(engine: Engine) -> list[ProgramState]:
    path = engine.get_corpus()
    files = glob.glob(f"{path}/*.js")
    logging.info(f"Found {len(files)} files in corpus")
    corpus: list[ProgramState] = []

    for file in tqdm.tqdm(files):
        exec_data_path = Path(file).with_suffix(".pkl")

        with open(file, "r") as f:
            code = f.read()

        if exec_data_path.exists():
            with open(exec_data_path, "rb") as f:
                exec_data = pickle.load(f)
        else:
            exec_data = engine.execute_text(code)
            with open(exec_data_path, "wb") as f:
                pickle.dump(exec_data, f)

        if exec_data is None or exec_data.error != JSError.NoError:
            logging.info(f"Failed to execute {file} or produced an error")
            continue

        try:
            ast = esprima.parseScript(code, tolerant=True, jsx=True)
            ast = Node.from_dict(ast.toDict())
            if not isinstance(ast, Node):
                logging.error(f"Failed to parse {file} when converting to custom ast")
                continue
            sanitise_ast(ast)

        except (esprima.error_handler.Error, UnknownNodeTypeError):  # type: ignore
            # logging.warning(f"Failed to parse {file}")
            continue

        corpus.append(ProgramState(ast, exec_data.coverage_data))  # type: ignore

    logging.info(f"Loaded {len(corpus)}/{len(files)} files from corpus")
    return corpus


def get_subtrees(corpus: list[ProgramState]) -> dict[str, list[Node]]:
    subtrees: dict[str, list[Node]] = defaultdict(list)

    for state in corpus:
        for node in state.current_node.traverse():
            if hasattr(node, "type"):
                subtrees[node.type].append(node)

    return subtrees


def sanitise_ast(ast: Node):
    for node in ast.traverse():
        for field in node.fields:
            val = getattr(node, field)
            if isinstance(val, list):
                new_body: list[Any] = []
                v: Any
                for v in val:
                    if (
                        isinstance(v, ExpressionStatement)
                        and isinstance(v.expression, CallExpression)
                        and v.expression.callee.name
                        and "assert" in v.expression.callee.name
                        # and "assert" in v.expression.callee.name
                    ):
                        continue

                    new_body.append(v)

                setattr(node, field, new_body)
