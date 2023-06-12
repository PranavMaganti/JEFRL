import copy
from enum import IntEnum
import logging
import os
from pathlib import Path
import pickle
import random
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
from rl.tokenizer import ASTTokenizer
import torch

from utils.js_engine import Coverage
from utils.js_engine import Engine
from utils.js_engine import ExecutionData

from transformers import RobertaModel


STATEMENT_PENALTY_WEIGHT = 3
COVERAGE_REWARD_WEIGHT = 2
MAX_STATEMENTS = 100


class FuzzingEnv(gym.Env[torch.Tensor, np.int64]):
    metadata = {}

    def __init__(
        self,
        corpus: list[ProgramState],
        subtrees: dict[str, list[Node]],
        engine: Engine,
        total_coverage: Coverage,
        ast_net: RobertaModel,
        tokenizer: ASTTokenizer,
        interesting_folder: Path,
        max_mutations: int = 25,
        max_statements: int = MAX_STATEMENTS,
        render_mode: Optional[str] = None,
    ):
        self.action_space = spaces.Discrete(len(FuzzingAction))
        # self.observation_space = spaces.Box() # type: ignore
        self.render_mode = render_mode

        self.corpus = corpus
        self.corpus_selection_count = [1 for _ in corpus]

        self.subtrees = subtrees
        self.engine = engine

        self.interesting_folder = interesting_folder
        os.makedirs(self.interesting_folder, exist_ok=True)

        self._state: ProgramState
        self.num_mutations = 0  # number of mutations performed
        self.max_mutations = max_mutations  # max number of mutations to perform
        self.max_statements = max_statements  # max number of statements in a program

        self.total_coverage = total_coverage
        self.coverage_increased = False  # whether coverage has increased

        self.total_executions = 0
        self.total_actions = 0

        self.ast_net = ast_net
        self.tokenizer = tokenizer

    def save_current_state(self, save_type: str, exec_data: ExecutionData) -> None:
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

    def _get_obs(self) -> torch.Tensor:
        tokenized_target = self.tokenizer.tokenize(self._state.get_target_node())
        tokenized_context = self.tokenizer.tokenize(self._state.get_context_node())

        batch = self.tokenizer.pad_batch([tokenized_target, tokenized_context])
        state = self.ast_net(**batch).pooler_output.view(1, -1)

        return state

    def _get_info(self) -> dict[str, str]:
        return {}

    def _get_reward(self, exec_data: Optional[ExecutionData] = None) -> float:
        num_statements = count_statements(self._state.program)
        penalty = min(0, 1 - num_statements / self.max_statements)

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
            self.coverage_increased = True
            logging.info(
                f"Coverage increased from {self.total_coverage} to {new_total_coverage}"
            )
            self.save_current_state("coverage", exec_data)
            self.corpus.append(self._state)
            self.corpus_selection_count.append(1)
            self.total_coverage = new_total_coverage

            return 2 + penalty
        elif exec_data.coverage.hit_edges > self._state.exec_data.coverage.hit_edges:
            # reward increasing coverage of test case but less than the reward for
            # increasing total coverage
            return 1 + penalty
        else:
            # new test case did not increase its own coverage
            return -1 + penalty

    def _get_done(self, exec_data: Optional[ExecutionData] = None) -> bool:
        return exec_data is not None and exec_data.is_crash()

    def _get_truncated(self) -> bool:
        return self.num_mutations >= self.max_mutations

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        # We need the following line to seed self.np_random
        super().reset(seed=seed, options=options)

        # We want to choose a program state with low selection count
        # program_state_counts = np.array(self.corpus_selection_count)
        # inverted_counts = (np.max(program_state_counts) + 1) - program_state_counts
        # weights = inverted_counts / np.sum(inverted_counts)
        # program_state_idx = self.np_random.choice(len(self.corpus), p=weights)
        program_state_idx = self.np_random.choice(len(self.corpus))
        self.corpus_selection_count[program_state_idx] += 1
        self._state = copy.deepcopy(self.corpus[program_state_idx])
        scope_analysis(self._state.program)

        # Initialise state as random child of the root node
        self._state.move_down()
        self.num_mutations = 0
        self.coverage_increased = False

        logging.info("Starting new episode")

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(
        self, action: np.int64
    ) -> tuple[torch.Tensor, float, bool, bool, dict[str, Any]]:
        logging.info(
            f"Number of mutations: {self.num_mutations}, action: {FuzzingAction(action)}"
        )
        self.total_actions += 1

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

        if not changed:
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
        code = self._state.generate_program_code()
        if not code:
            # Negative reward for invalid program
            return (
                self._get_obs(),
                -1,
                self._get_truncated(),
                self._get_done(),
                self._get_info(),
            )
        exec_data = self.engine.execute_text(code)
        self.total_executions += 1

        if not exec_data:
            # Negative reward for invalid program
            return self._get_obs(), -1, self._get_truncated(), True, self._get_info()

        reward = self._get_reward(exec_data)
        done = self._get_done(exec_data)

        self._state.exec_data = exec_data
        return self._get_obs(), reward, self._get_truncated(), done, self._get_info()

    def _move(
        self, action: FuzzingAction
    ) -> tuple[torch.Tensor, float, bool, bool, dict[str, Any]]:
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

        return (
            self._get_obs(),
            self._get_reward() if moved else -1,
            self._get_truncated(),
            self._get_done(),
            self._get_info(),
        )

    def _end(
        self,
    ) -> tuple[torch.Tensor, float, bool, bool, dict[str, Any]]:
        return (
            self._get_obs(),
            0 if self.coverage_increased else -2,
            True,
            self._get_done(),
            self._get_info(),
        )
