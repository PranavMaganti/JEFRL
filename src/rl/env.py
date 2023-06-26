import copy
import logging
from pathlib import Path
import pickle
import time
from typing import Any, Optional

import gymnasium as gym
from gymnasium import spaces
from js_ast.analysis import count_statements
from js_ast.analysis import scope_analysis
from js_ast.nodes import Node
import numpy as np
from rl.fuzzing_action import FuzzingAction
from rl.program_state import ProgramState
from transformer.tokenizer import ASTTokenizer

from utils.js_engine import Coverage
from utils.js_engine import Engine
from utils.js_engine import ExecutionData


PENALTY_STATEMENTS = 100
MAX_STATEMENTS = 1000
MAX_FRAGMENT_SEQ_LEN = 512  # Maximum length of the AST fragment sequence


class FuzzingEnv(gym.Env[tuple[list[int], list[int]], np.int64]):
    metadata = {}

    def __init__(
        self,
        corpus: list[ProgramState],
        subtrees: dict[str, list[Node]],
        engine: Engine,
        total_coverage: Coverage,
        tokenizer: ASTTokenizer,
        interesting_folder: Optional[Path] = None,
        max_mutations: int = 50,
        penalty_statements: int = PENALTY_STATEMENTS,
        max_statements: int = MAX_STATEMENTS,
        max_eps_without_coverage_increase: int = 500,
        render_mode: Optional[str] = None,
    ):
        self.action_space = spaces.Discrete(len(FuzzingAction))
        self.observation_space = spaces.Tuple(
            (
                spaces.Box(
                    low=0,
                    high=len(tokenizer.vocab) - 1,
                    shape=(MAX_FRAGMENT_SEQ_LEN,),
                    dtype=np.int64,
                ),
                spaces.Box(
                    low=0,
                    high=len(tokenizer.vocab) - 1,
                    shape=(MAX_FRAGMENT_SEQ_LEN,),
                    dtype=np.int64,
                ),
            )
        )
        self.render_mode = render_mode

        self.corpus = corpus

        # Stores number of times each program in the corpus has been selected
        # since last coverage increase
        self.corpus_selection_count = [0] * len(corpus)

        self.subtrees = subtrees
        self.engine = engine

        self.interesting_folder = interesting_folder

        self._state: ProgramState
        self._state_idx: int

        self.num_mutations = 0  # number of mutations performed
        self.max_mutations = max_mutations  # max number of mutations to perform
        self.penalty_statements = (
            penalty_statements  # number of statements after which to apply penalty
        )
        self.max_statements = max_statements  # max number of statements in a program
        self.max_eps_without_coverage_increase = max_eps_without_coverage_increase

        self.total_coverage = total_coverage
        self.test_coverage_increased = False  # whether coverage has increased
        self.total_coverage_increased = False  # whether coverage has increased

        self.total_executions = 0
        self.total_actions = 0
        self.failed_actions = []

        self.exec_errors = []

        self.tokenizer = tokenizer

        self.exec_times = []
        self.action_times = []
        self.code_gen_times = []

    def save_current_state(self, save_type: str, exec_data: ExecutionData) -> None:
        if self.interesting_folder is None:
            return

        current_time = int(time.time())

        code = self._state.generate_program_code()
        if code is None:
            return

        with open(self.interesting_folder / f"{current_time}_{save_type}.js", "w") as f:
            f.write(code)

        with open(
            self.interesting_folder / f"{current_time}_{save_type}.ps", "wb"
        ) as f:
            pickle.dump(self._state, f)

        with open(
            self.interesting_folder / f"{current_time}_{save_type}.txt", "w"
        ) as f:
            f.write(exec_data.out)

    def _get_obs(self) -> tuple[list[int], list[int]]:
        tokenized_target = self.tokenizer.tokenize(self._state.get_target_node())
        tokenized_context = self.tokenizer.tokenize(self._state.get_context_node())

        return (tokenized_target, tokenized_context)

    def _get_info(self) -> dict[str, str]:
        return {}

    def _get_reward(self, exec_data: Optional[ExecutionData] = None) -> float:
        num_statements = count_statements(self._state.root)
        penalty = min(0, 1 - num_statements / self.penalty_statements)

        if exec_data is None:
            return penalty

        new_total_coverage = exec_data.coverage | self.total_coverage

        if exec_data.is_crash():
            self.total_coverage = new_total_coverage
            logging.info(f"Crash detected: {exec_data.out}")
            self.save_current_state("crash", exec_data)
            return 3 + penalty

        if new_total_coverage != self.total_coverage:
            # new test case increased its own coverage and the total coverage
            self.total_coverage_increased = True
            self.test_coverage_increased = True

            logging.info(
                f"Coverage increased from {self.total_coverage} to {new_total_coverage}"
            )
            self.save_current_state("coverage", exec_data)
            self.corpus.append(self._state)
            self.corpus_selection_count.append(0)
            self.total_coverage = new_total_coverage

            return 2 + penalty
        elif exec_data.coverage.hit_edges > self._state.exec_data.coverage.hit_edges:
            # reward increasing coverage of test case but less than the reward for
            # increasing total coverage
            self.test_coverage_increased = True
            return 1 + penalty
        elif exec_data.coverage.hit_edges < self._state.exec_data.coverage.hit_edges:
            if self._get_truncated() and not self.total_coverage_increased:
                # episode is over and coverage did not increase
                return -2 + penalty

            # reward decreasing coverage of test case
            return -0.1 + penalty
        else:
            if self._get_truncated() and not self.total_coverage_increased:
                # episode is over and coverage did not increase
                return -2 + penalty

            # new test case did not increase its own coverage
            return 0 + penalty

    def _get_done(self, exec_data: Optional[ExecutionData] = None) -> bool:
        return exec_data is not None and exec_data.is_crash()

    def _get_truncated(self) -> bool:
        num_statements = count_statements(self._state.root)
        return (
            self.num_mutations >= self.max_mutations
            or num_statements >= self.max_statements
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[tuple[list[int], list[int]], dict[str, Any]]:
        # We need the following line to seed self.np_random
        super().reset(seed=seed, options=options)
        if hasattr(self, "_state"):
            if self.total_coverage_increased:
                self.corpus_selection_count[self._state_idx] = 0
            else:
                self.corpus_selection_count[self._state_idx] += 1
                if (
                    self.corpus_selection_count[self._state_idx]
                    >= self.max_eps_without_coverage_increase
                ):
                    self.corpus.pop(self._state_idx)
                    self.corpus_selection_count.pop(self._state_idx)

                    logging.info(
                        f"Removed program from corpus: {self._state.generate_program_code()}"
                    )

        # We want to choose a program state with low selection count
        # program_state_counts = np.array(self.corpus_selection_count)
        # inverted_counts = (np.max(program_state_counts) + 1) - program_state_counts
        # weights = inverted_counts / np.sum(inverted_counts)
        # program_state_idx = self.np_random.choice(len(self.corpus), p=weights)
        self._state_idx = self.np_random.choice(len(self.corpus))
        self._state = copy.deepcopy(self.corpus[self._state_idx])

        scope_analysis(self._state.root)

        # Initialise state as random child of the root node
        self._state.move_down()
        self.num_mutations = 0
        self.total_coverage_increased = False
        self.test_coverage_increased = False

        logging.info("Starting new episode")

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(
        self, action: np.int64
    ) -> tuple[tuple[list[int], list[int]], float, bool, bool, dict[str, Any]]:
        logging.info(
            f"Number of mutations: {self.num_mutations}, action: {FuzzingAction(action)}"
        )
        self.total_actions += 1
        start_time = time.time()
        match (action):
            case FuzzingAction.MOVE_UP | FuzzingAction.MOVE_DOWN | FuzzingAction.MOVE_LEFT | FuzzingAction.MOVE_RIGHT:
                return self._move(action)
            case FuzzingAction.END:
                return self._end()
            case FuzzingAction.REPLACE:
                new_node, changed = self._state.replace(self.subtrees)
            case FuzzingAction.ADD:
                new_node, changed = self._state.add(self.subtrees)
            case FuzzingAction.REMOVE:
                new_node, changed = self._state.remove(self.subtrees)
            case FuzzingAction.MODIFY:
                changed = self._state.modify()
                new_node = self._state.target_node
            case _:
                raise ValueError(f"Invalid action: {action}")
        end_time = time.time()
        logging.info(f"Action time: {end_time - start_time} seconds")
        self.action_times.append(end_time - start_time)

        if not changed:
            self.failed_actions.append(
                (action, self._state.target_node.type, self.total_actions)
            )
            # Negative reward for action which does not change the state
            return (
                self._get_obs(),
                -1,
                self._get_truncated(),
                self._get_done(),
                self._get_info(),
            )

        self.num_mutations += 1
        self._state.target_node = new_node
        start = time.time()
        code = self._state.generate_program_code()
        end = time.time()
        self.code_gen_times.append(end - start)
        logging.info(f"Code generation time: {end - start} seconds")

        if not code:
            # Negative reward for invalid program
            return (
                self._get_obs(),
                -1,
                self._get_truncated(),
                self._get_done(),
                self._get_info(),
            )

        start = time.time()
        exec_data = self.engine.execute_text(code)
        end = time.time()
        self.exec_times.append(end - start)
        logging.info(f"Execution time: {end - start} seconds")

        self.total_executions += 1

        if not exec_data:
            # Negative reward for invalid program
            return self._get_obs(), -1, self._get_truncated(), True, self._get_info()

        self.exec_errors.append(exec_data.error)

        reward = self._get_reward(exec_data)
        done = self._get_done(exec_data)

        self._state.exec_data = exec_data
        return self._get_obs(), reward, self._get_truncated(), done, self._get_info()

    def _move(
        self, action: FuzzingAction
    ) -> tuple[tuple[list[int], list[int]], float, bool, bool, dict[str, Any]]:
        match (action):
            case FuzzingAction.MOVE_UP:
                moved = self._state.move_up()
            case FuzzingAction.MOVE_DOWN:
                moved = self._state.move_down()
            case FuzzingAction.MOVE_LEFT:
                moved = self._state.move_left()
            case FuzzingAction.MOVE_RIGHT:
                moved = self._state.move_right()
            case _:
                raise ValueError(f"Invalid action: {action}")

        if not moved:
            self.failed_actions.append((action, self._state.target_node.type))

        return (
            self._get_obs(),
            0 if moved else -1,
            self._get_truncated(),
            self._get_done(),
            self._get_info(),
        )

    def _end(
        self,
    ) -> tuple[tuple[list[int], list[int]], float, bool, bool, dict[str, Any]]:
        return (
            self._get_obs(),
            0 if self.total_coverage_increased else -2,
            True,
            self._get_done(),
            self._get_info(),
        )
