from collections import Counter
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import copy
from dataclasses import dataclass
import logging
from pathlib import Path
import pickle
import queue
from typing import Optional

import esprima
from js_ast.analysis import count_statements
from js_ast.natives import encode_natives
from js_ast.nodes import Node
from js_ast.nodes import UnknownNodeTypeError
from preprocessing.normalise import normalize_ast
from preprocessing.sanitise import sanitise_ast
from rl.env import MAX_STATEMENTS
from rl.program_state import ProgramState
import tqdm

from utils.js_engine import Coverage
from utils.js_engine import Engine
from utils.js_engine import JSError


# Parsed ASTs are cached next to each corpus file. The suffix is versioned so caches
# written by older versions of the loader (which dropped natives syntax) are ignored.
AST_CACHE_SUFFIX = ".jefrl-v2.ast"


def read_code(file: Path) -> Optional[str]:
    try:
        with open(file, "r", encoding="utf-8") as f:
            return f.read()
    except (UnicodeDecodeError, OSError):
        return None


def load_raw_corpus(corpus_path: Path) -> list[Node]:
    asts: list[Node] = []

    files = list(corpus_path.rglob("*.js"))
    for file in tqdm.tqdm(files):
        code = read_code(file)
        if code is None:
            continue

        ast_path = Path(file).with_suffix(AST_CACHE_SUFFIX)
        ast = load_ast(code, file, ast_path)

        if not isinstance(ast, Node):
            logging.warning(f"Failed to parse {file} when converting to ast")
            continue

        asts.append(ast)

    return asts


# Harness libraries to try for a seed, cheapest to load first. Many seeds need no
# harness once assertions are removed, and some test suites use helpers from
# another engine's shell (e.g. JavaScriptCore's noInline in the jit folder).
HARNESS_LIBS = [
    Path("corpus/libs/chakracore.js"),
    Path("corpus/libs/javascriptcore.js"),
    Path("corpus/libs/spidermonkey.js"),
    # Combined harness defining the helpers of every engine, but slow to load
    Path("corpus/libs/v8.js"),
]


@dataclass
class SeedResult:
    state: Optional[ProgramState] = None
    status: str = "ok"
    crash_report: str = ""


def _candidate_libs(folder_lib: Optional[Path]) -> list[Optional[Path]]:
    libs: list[Optional[Path]] = [None]
    for lib in HARNESS_LIBS[:2] + [folder_lib] + HARNESS_LIBS[2:]:
        if lib is not None and lib not in libs and lib.exists():
            libs.append(lib)
    return libs


def load_seed(engine: Engine, file: Path, folder_lib: Optional[Path]) -> SeedResult:
    """Parses and cleans a seed, then finds a way to run it without errors: first
    with identifiers normalised (better for the Transformer's vocabulary), then
    without, each with no harness and then with increasingly large harnesses."""
    code = read_code(file)
    if code is None:
        return SeedResult(status="read")

    ast = load_ast(code, file, Path(file).with_suffix(AST_CACHE_SUFFIX))
    if ast is None:
        return SeedResult(status="parse")

    sanitise_ast(ast)
    try:
        normalised = copy.deepcopy(ast)
    except RecursionError:
        # Too deeply nested for the fuzzer, which copies the program every episode
        return SeedResult(status="parse")
    normalize_ast(normalised)

    status = "codegen"
    for variant in (normalised, ast):
        state = ProgramState(variant, None)  # type: ignore
        program_code = state.generate_program_code()
        if program_code is None:
            continue

        for lib in _candidate_libs(folder_lib):
            exec_data = engine.execute_text(program_code, lib)
            if exec_data is None:
                status = "exec"
                continue

            if exec_data.is_crash():
                # Seeds which crash the engine are most likely known bugs (e.g. the
                # PoCs in js-vuln-db). They are not fuzzed, but are useful to check
                # that crash detection works on the engine under test.
                return SeedResult(
                    status="crash",
                    crash_report=f"{file}\n{exec_data.err.strip()[:2000]}\n",
                )

            if exec_data.error == JSError.TimeoutError:
                # A harness won't make it faster
                return SeedResult(status="timeout")

            if exec_data.error == JSError.NoError:
                state.lib_path = lib
                state.exec_data = exec_data
                return SeedResult(state)

            status = "exec"

    return SeedResult(status=status)


def load_corpus(
    engines: Engine | list[Engine],
    corpus_path: Path,
    corpus_folders: list[dict[str, str]],
    crash_log: Optional[Path] = None,
) -> list[ProgramState]:
    """Loads, sanitises and normalises every seed, then executes the resulting code
    and keeps the seeds which run cleanly (see load_seed).

    The code that is executed is generated from the processed AST, so the stored
    coverage matches what the fuzzer will execute when mutating the seed. With
    several engines, seeds are executed in parallel.
    """
    if not isinstance(engines, list):
        engines = [engines]

    idle: queue.Queue[Engine] = queue.Queue()
    for engine in engines:
        idle.put(engine)

    def process(args: tuple[Path, Optional[Path]]) -> SeedResult:
        engine = idle.get()
        try:
            return load_seed(engine, *args)
        finally:
            idle.put(engine)

    corpus: list[ProgramState] = []
    crashes: list[str] = []
    statuses: Counter[str] = Counter()
    lib_counts: Counter[str] = Counter()

    with ThreadPoolExecutor(max_workers=len(engines)) as pool:
        for item in corpus_folders:
            folder_path = corpus_path / item["folder_name"]
            folder_lib = Path(item["lib_path"]) if item["lib_path"] else None

            files = sorted(folder_path.rglob("*.js"))
            logging.info(f"Found {len(files)} files in corpus folder {item['folder_name']}")

            results = pool.map(process, [(file, folder_lib) for file in files])
            for result in tqdm.tqdm(
                results, total=len(files), desc=f"Loading {item['folder_name']}"
            ):
                statuses[result.status] += 1
                if result.state is not None:
                    corpus.append(result.state)
                    lib = result.state.lib_path
                    lib_counts[lib.name if lib else "none"] += 1
                if result.crash_report:
                    logging.warning(f"Seed crashed the engine: {result.crash_report.splitlines()[0]}")
                    crashes.append(result.crash_report)

    logging.info(f"Failed to read {statuses['read']} files")
    logging.info(f"Failed to parse {statuses['parse']} files")
    logging.info(f"Failed to generate code for {statuses['codegen']} files")
    logging.info(f"Timed out: {statuses['timeout']} files")
    logging.info(f"Threw an error with every harness: {statuses['exec']} files")
    logging.info(f"Seeds which crashed the engine: {len(crashes)}")
    logging.info(f"Harness used by loaded seeds: {dict(lib_counts)}")
    logging.info(f"Loaded {len(corpus)} files from corpus")

    if crash_log is not None:
        crash_log.write_text("\n".join(crashes))

    return corpus


def parse_code(code: str, file_name: str) -> Optional[Node]:
    try:
        ast = esprima.parseScript(encode_natives(code), tolerant=True, jsx=True)
        ast = Node.from_dict(ast.toDict(), file_name)
    except (esprima.error_handler.Error, UnknownNodeTypeError, RecursionError):  # type: ignore
        return None

    return ast if isinstance(ast, Node) else None


def load_ast(code: str, code_path: Path, ast_path: Path) -> Optional[Node]:
    if ast_path.exists():
        try:
            with open(ast_path, "rb") as f:
                return pickle.load(f)
        except (EOFError, pickle.UnpicklingError):
            # Truncated cache file (e.g. from an interrupted run), parse again
            pass

    ast = parse_code(code, code_path.name)
    try:
        data = pickle.dumps(ast)
    except RecursionError:
        # Too deeply nested to pickle, which the corpus and deep copies need too
        ast = None
        data = pickle.dumps(None)
    with open(ast_path, "wb") as f:
        f.write(data)

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
