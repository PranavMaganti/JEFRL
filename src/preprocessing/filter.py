from typing import Tuple

from js_ast.analysis import count_statements
from rl.env import MAX_STATEMENTS
from rl.program_state import ProgramState
import tqdm

from utils.js_engine import Coverage


def filter_corpus_by_coverage(
    corpus: list[ProgramState],
) -> Tuple[list[ProgramState], Coverage]:
    filtered_corpus: list[ProgramState] = []
    total_coverage = Coverage()

    for state in tqdm.tqdm(corpus, desc="Filtering corpus by coverage"):
        new_coverage = total_coverage | state.exec_data.coverage

        if new_coverage != total_coverage:
            filtered_corpus.append(state)
            total_coverage = new_coverage

    return filtered_corpus, total_coverage


def filter_corpus_by_length(
    corpus: list[ProgramState], max_statements: int = MAX_STATEMENTS
) -> list[ProgramState]:
    filtered_corpus: list[ProgramState] = []

    for state in tqdm.tqdm(corpus, desc="Filtering corpus by length"):
        if count_statements(state.root) <= max_statements:
            filtered_corpus.append(state)

    return filtered_corpus
