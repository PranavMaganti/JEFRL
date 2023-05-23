from collections import defaultdict
import glob
import logging
from pathlib import Path
import pickle
from typing import Any, Optional

import esprima
from js_ast.nodes import CallExpression
from js_ast.nodes import ExpressionStatement
from js_ast.nodes import Node
from js_ast.nodes import UnknownNodeTypeError
from rl.program_state import ProgramState
import tqdm

from utils.js_engine import Coverage
from utils.js_engine import Engine
from utils.js_engine import ExecutionData
from utils.js_engine import JSError


# Loads corpus, skipping files that do not increase the coverage
def load_corpus(engine: Engine) -> tuple[list[ProgramState], Coverage]:
    files = glob.glob(f"{engine.corpus_path}/**/*.js")
    logging.info(f"Found {len(files)} files in corpus")
    corpus: list[ProgramState] = []
    total_coverage = Coverage()

    failed_exec = 0
    failed_parse = 0
    non_increasing_coverage = 0

    for file in tqdm.tqdm(files):
        exec_data_path = Path(file).with_suffix(".pkl")
        ast_path = Path(file).with_suffix(".ast")

        with open(file, "r") as f:
            code = f.read()

        exec_data = load_exec_data(code, engine, exec_data_path)

        if exec_data is None or exec_data.error != JSError.NoError:
            logging.warning(f"Failed to execute {file} or produced an error")
            failed_exec += 1
            continue

        new_coverage = exec_data.coverage | total_coverage

        if new_coverage == total_coverage:
            logging.warning(f"Skipping {file} as it does not increase coverage")
            non_increasing_coverage += 1
            continue

        total_coverage = new_coverage

        ast = load_ast(code, ast_path)

        if ast is None:
            logging.warning(f"Failed to parse {file} when converting to ast")
            failed_parse += 1
            continue

        corpus.append(ProgramState(ast, exec_data.coverage, file))

    logging.info(f"Failed to execute {failed_exec} files")
    logging.info(f"Failed to parse {failed_parse} files")
    logging.info(
        f"Skipped {non_increasing_coverage} files due to non-increasing coverage"
    )
    logging.info(f"Loaded {len(corpus)}/{len(files)} files from corpus")

    return corpus, total_coverage


def load_exec_data(code: str, engine: Engine, path: Path) -> Optional[ExecutionData]:
    if path.exists():
        with open(path, "rb") as f:
            exec_data = pickle.load(f)
    else:
        exec_data = engine.execute_text(code)
        with open(path, "wb") as f:
            pickle.dump(exec_data, f)

    return exec_data


def load_ast(code: str, path: Path) -> Optional[Node]:
    if path.exists():
        with open(path, "rb") as f:
            ast = pickle.load(f)
    else:
        try:
            ast = esprima.parseScript(code, tolerant=True, jsx=True)
            ast = Node.from_dict(ast.toDict())
            if not isinstance(ast, Node):
                return None
            sanitise_ast(ast)
            with open(path, "wb") as f:
                pickle.dump(ast, f)

        except (esprima.error_handler.Error, UnknownNodeTypeError):  # type: ignore
            # logging.warning(f"Failed to parse {file}")
            return None

    return ast


def get_subtrees(corpus: list[ProgramState]) -> dict[str, list[Node]]:
    subtrees: dict[str, list[Node]] = defaultdict(list)

    for state in corpus:
        for node in state.target_node.traverse():
            if hasattr(node, "type"):
                subtrees[node.type].append(node)

    return subtrees


# Remove all assert statements from the AST
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
