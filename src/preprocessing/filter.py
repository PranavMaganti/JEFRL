from typing import Tuple

from rl.program_state import ProgramState
import tqdm

from utils.js_engine import Coverage


def filter_corpus_by_coverage(
    corpus: list[ProgramState],
) -> Tuple[list[ProgramState], Coverage]:
    filtered_corpus: list[ProgramState] = []
    total_coverage = Coverage()

    for state in tqdm.tqdm(corpus, desc="Filtering corpus"):
        new_coverage = total_coverage | state.exec_data.coverage

        if new_coverage != total_coverage:
            filtered_corpus.append(state)
            total_coverage = new_coverage

    return filtered_corpus, total_coverage
