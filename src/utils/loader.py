from collections import defaultdict
import glob
import logging
from pathlib import Path
import pickle
from typing import Optional

import esprima
from js_ast.analysis import count_statements
from js_ast.nodes import Node
from js_ast.nodes import UnknownNodeTypeError
from preprocessing.sanitise import sanitise_ast
from rl.env import MAX_STATEMENTS
from rl.program_state import ProgramState
import tqdm

from utils.js_engine import Coverage
from utils.js_engine import Engine
from utils.js_engine import ExecutionData
from utils.js_engine import JSError


def load_raw_corpus(corpus_path: Path) -> list[Node]:
    asts: list[Node] = []

    files = list(corpus_path.rglob("*.js"))
    for file in tqdm.tqdm(files):
        with open(file, "r") as f:
            code = f.read()

        ast_path = Path(file).with_suffix(".ast")
        ast = load_ast(code, file, ast_path)

        if not isinstance(ast, Node):
            logging.warning(f"Failed to parse {file} when converting to ast")
            continue

        asts.append(ast)

    return asts


# Loads corpus, skipping files that do not increase the coverage
def load_corpus(
    engine: Engine, corpus_path: Path, corpus_folders: list[dict[str, str]]
) -> list[ProgramState]:
    corpus: list[ProgramState] = []

    failed_exec = 0
    failed_parse = 0

    for item in corpus_folders:
        folder_path = corpus_path / item["folder_name"]
        lib_path = Path(item["lib_path"]) if item["lib_path"] else None

        files = list(folder_path.rglob("*.js"))
        logging.info(f"Found {len(files)} files in corpus folder {item['folder_name']}")

        for file in tqdm.tqdm(files, desc="Loading corpus"):
            # print(file)
            exec_data_path = Path(file).with_suffix(".pkl")
            ast_path = Path(file).with_suffix(".ast")

            with open(file, "r") as f:
                code = f.read()

            exec_data = load_exec_data(code, engine, exec_data_path)

            if exec_data is None or exec_data.error != JSError.NoError:
                logging.warning(f"Failed to execute {file} or produced an error")
                failed_exec += 1
                continue

            ast = load_ast(code, file, ast_path)

            if ast is None:
                logging.warning(f"Failed to parse {file} when converting to ast")
                failed_parse += 1
                continue

            corpus.append(ProgramState(ast, exec_data, lib_path))

    logging.info(f"Failed to execute {failed_exec} files")
    logging.info(f"Failed to parse {failed_parse} files")
    logging.info(f"Loaded {len(corpus)} files from corpus")

    return corpus


def load_exec_data(code: str, engine: Engine, path: Path) -> Optional[ExecutionData]:
    if path.exists():
        with open(path, "rb") as f:
            exec_data = pickle.load(f)
    else:
        exec_data = engine.execute_text(code)
        with open(path, "wb") as f:
            pickle.dump(exec_data, f)

    return exec_data


def load_ast(code: str, code_path: Path, ast_path: Path) -> Optional[Node]:
    if ast_path.exists():
        with open(ast_path, "rb") as f:
            ast = pickle.load(f)
    else:
        try:
            ast = esprima.parseScript(code, tolerant=True, jsx=True)
            ast = Node.from_dict(ast.toDict(), code_path.name)
            if not isinstance(ast, Node):
                return None

            with open(ast_path, "wb") as f:
                pickle.dump(ast, f)

        except (esprima.error_handler.Error, UnknownNodeTypeError, RecursionError):  # type: ignore
            # logging.warning(f"Failed to parse {file}")
            with open(ast_path, "wb") as f:
                pickle.dump(None, f)
            return None

    return ast


def get_subtrees(states: list[ProgramState]) -> dict[str, list[Node]]:
    subtrees: dict[str, list[Node]] = defaultdict(list)

    for state in tqdm.tqdm(states, desc="Collecting subtrees"):
        for node in state.target_node.traverse():
            if hasattr(node, "type"):
                statement_count = count_statements(node)

                if statement_count < MAX_STATEMENTS:
                    subtrees[node.type].append(node)

    return subtrees


def load_program_states(folder: Path) -> tuple[list[ProgramState], Coverage]:
    programs = []
    total_coverage = Coverage()

    files = list(folder.rglob("*.ps"))
    for file in tqdm.tqdm(files, desc="Loading resume program states"):
        with open(file, "rb") as f:
            program = pickle.load(f)

        if isinstance(program, ProgramState):
            programs.append(program)
            total_coverage |= program.exec_data.coverage
        else:
            print("Failed to load program state")

    return programs, total_coverage
