import logging
import random
import time
from enum import IntEnum
from functools import reduce
from pathlib import Path
from typing import Any, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from tqdm import tqdm

from js_ast import escodegen
from js_ast.mutation import add, remove, replace
from js_ast.nodes import Node
from utils.js_engine import CoverageData, Engine, ExecutionData

INTERESTING_FOLDER = Path("corpus/interesting")


class FuzzingAction(IntEnum):
    REPLACE = 0
    ADD = 1
    REMOVE = 2
    MOVE_UP = 3
    MOVE_DOWN = 4
    END = 5

    def __str__(self):
        match self:
            case FuzzingAction.REPLACE:
                return "Replace"
            case FuzzingAction.ADD:
                return "Add"
            case FuzzingAction.REMOVE:
                return "Remove"
            case FuzzingAction.MOVE_UP:
                return "Move Up"
            case FuzzingAction.MOVE_DOWN:
                return "Move Down"
            case FuzzingAction.END:
                return "End"


class ProgramState:
    def __init__(self, program: Node, coverage_data: CoverageData, original_file: str):
        self.program = program
        self.coverage_data = coverage_data
        self.target_node = program
        self.context_node = program
        self.original_file = original_file
        self.history = []

    def generate_code(self, node: Node) -> str:
        try:
            return escodegen.generate(node)  # type: ignore
        except Exception:
            logging.error("Error generating code")
            return ""

    def generate_target_code(self) -> str:
        return self.generate_code(self.target_node)

    def generate_context_code(self) -> str:
        return self.generate_code(self.context_node)

    def generate_program_code(self) -> str:
        return self.generate_code(self.program)

    def __str__(self):
        return self.__repr__()


class FuzzingEnv(gym.Env[str, np.int64]):
    metadata = {}

    def __init__(
        self,
        corpus: list[ProgramState],
        subtrees: dict[str, list[Node]],
        max_mutations: int,
        engine: Engine,
        render_mode: Optional[str] = None,
    ):
        self.action_space = spaces.Discrete(len(FuzzingAction))
        self.observation_space = spaces.Text(max_length=10000)
        self.render_mode = render_mode

        self.corpus = corpus
        self.subtrees = subtrees
        self.engine = engine

        self._state: ProgramState
        self.num_mutations = 0  # number of mutations performed
        self.max_mutations = max_mutations  # max number of mutations to perform
        self.current_coverage = CoverageData()
        for state in tqdm(corpus):
            self.current_coverage |= state.coverage_data

    def save_current_state(self, path: Path):
        with open(path, "w") as f:
            f.write(self._state.generate_program_code())

    def _get_obs(self) -> str:
        obs = self._state.generate_target_code()
        return obs if obs else ""

    def _get_info(self) -> dict[str, str]:
        return {}

    def _get_reward(self, exec_data: ExecutionData):
        if exec_data.is_crash():
            print(f"Crash detected: {exec_data.out}")
            self.save_current_state(INTERESTING_FOLDER / f"{time.time()}_crash.js")
            return 10

        new_coverage = exec_data.coverage_data | self.current_coverage

        # new coverage is the same as the current coverage
        if new_coverage == self.current_coverage:
            return 0

        # new coverage has increased total coverage
        print(f"Coverage increased from {self.current_coverage} to {new_coverage}")
        self.save_current_state(INTERESTING_FOLDER / f"{time.time()}.js")
        self.corpus.append(self._state)
        self.current_coverage = new_coverage

        return 5

    def _get_done(self, exec_data: Optional[ExecutionData] = None) -> bool:
        return exec_data is not None and exec_data.is_crash()

    def _get_truncated(self) -> bool:
        return self.num_mutations >= self.max_mutations

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[str, dict[str, Any]]:
        # We need the following line to seed self.np_random
        super().reset(seed=seed, options=options)

        # Choose the agent's location uniformly at random
        self._state = random.choice(self.corpus)
        self.num_mutations = 0

        print("Starting new episode")

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, action: np.int64) -> tuple[str, float, bool, bool, dict[str, Any]]:
        print(
            f"Number of mutations: {self.num_mutations}, action: {FuzzingAction(action)}"
        )
        new_node = self._state.target_node

        match (action):
            case FuzzingAction.MOVE_UP:
                return self._move_up()
            case FuzzingAction.MOVE_DOWN:
                return self._move_down()
            case FuzzingAction.END:
                return self._end()
            case FuzzingAction.REPLACE:
                new_node = self._replace()
            case FuzzingAction.ADD:
                new_node = self._add()
            case FuzzingAction.REMOVE:
                new_node = self._remove()
            case _:
                raise ValueError(f"Invalid action: {action}")

        self.num_mutations += 1
        if new_node is self._state.target_node:
            return (
                self._get_obs(),
                0,
                self._get_truncated(),
                self._get_done(),
                self._get_info(),
            )

        self._state.target_node = new_node
        exec_data = self.engine.execute_text(self._state.generate_program_code())
        if not exec_data:
            return self._get_obs(), 0, self._get_truncated(), True, self._get_info()

        self._state.coverage_data = exec_data.coverage_data
        reward = self._get_reward(exec_data)
        done = self._get_done(exec_data)

        return self._get_obs(), reward, self._get_truncated(), done, self._get_info()

    def _move_up(self) -> tuple[str, float, bool, bool, dict[str, Any]]:
        if self._state.target_node.parent is None:
            return (
                self._get_obs(),
                -1,
                self._get_truncated(),
                self._get_done(),
                self._get_info(),
            )

        self._state.target_node = self._state.target_node.parent
        return (
            self._get_obs(),
            0,
            self._get_truncated(),
            self._get_done(),
            self._get_info(),
        )

    def _move_down(self) -> tuple[str, float, bool, bool, dict[str, Any]]:
        children = self._state.target_node.children()
        if children:
            self._state.target_node = random.choice(children)

        return (
            self._get_obs(),
            0,
            self._get_truncated(),
            self._get_done(),
            self._get_info(),
        )

    def _end(self) -> tuple[str, float, bool, bool, dict[str, Any]]:
        return self._get_obs(), 0, True, self._get_done(), self._get_info()

    def _replace(self) -> Node:
        return replace(self.subtrees, self._state.target_node)

    def _add(self) -> Node:
        return add(self.subtrees, self._state.target_node)

    def _remove(self) -> Node:
        return remove(self._state.target_node)
