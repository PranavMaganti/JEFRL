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


class ProgramState:
    def __init__(self, program: Node, coverage_data: CoverageData):
        self.program = program
        self.coverage_data = coverage_data
        self.current_node = program

    def generate_node_code(self) -> str:
        try:
            return escodegen.generate(self.current_node)  # type: ignore
        except Exception:
            logging.error("Error generating code")
            return ""

    def generate_program_code(self) -> str:
        try:
            return escodegen.generate(self.program)  # type: ignore
        except Exception:
            logging.error("Error generating code")
            return ""

    def __str__(self):
        return self.__repr__()


class FuzzingEnv(gym.Env[str, np.int64]):
    metadata = {}

    def __init__(
        self,
        corpus: list[ProgramState],
        subtrees: dict[str, list[Node]],
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
        self.steps_since_increased_coverage = 0
        self.current_coverage = reduce(
            lambda x, y: x | y.coverage_data, corpus, CoverageData()
        )

    def save_current_state(self, path: Path):
        with open(path, "w") as f:
            f.write(self._state.generate_program_code())

    def _get_obs(self) -> str:
        obs = self._state.generate_node_code()
        return obs if obs else ""

    def _get_info(self) -> dict[str, str]:
        return {}

    def _get_reward(self, exec_data: ExecutionData):
        if exec_data.is_crash():
            print(f"Crash detected: {exec_data.out}")
            self.save_current_state(INTERESTING_FOLDER / f"{time.time()}_crash.js")
            self.steps_since_increased_coverage = 0
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
        self.steps_since_increased_coverage = 0

        return 5

    def _get_done(self, exec_data: Optional[ExecutionData] = None) -> bool:
        return exec_data is not None and exec_data.is_crash()

    def _get_truncated(self) -> bool:
        return self.steps_since_increased_coverage > 500

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
        self.steps_since_increased_coverage = 0

        print("Starting new episode")

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, action: np.int64) -> tuple[str, float, bool, bool, dict[str, Any]]:
        print(
            f"Steps since increased coverage: {self.steps_since_increased_coverage}, action: {action}"
        )
        self.steps_since_increased_coverage += 1
        new_node = self._state.current_node

        if action == FuzzingAction.MOVE_UP:
            return self._move_up()
        elif action == FuzzingAction.MOVE_DOWN:
            return self._move_down()
        elif action == FuzzingAction.END:
            return self._end()
        elif action == FuzzingAction.REPLACE:
            new_node = self._replace()
        elif action == FuzzingAction.ADD:
            new_node = self._add()
        elif action == FuzzingAction.REMOVE:
            new_node = self._remove()
        else:
            raise ValueError(f"Invalid action: {action}")

        if new_node is self._state.current_node:
            return (
                self._get_obs(),
                0,
                self._get_truncated(),
                self._get_done(),
                self._get_info(),
            )

        self._state.current_node = new_node
        exec_data = self.engine.execute_text(self._state.generate_program_code())
        if not exec_data:
            return self._get_obs(), 0, self._get_truncated(), True, self._get_info()

        self._state.coverage_data = exec_data.coverage_data
        reward = self._get_reward(exec_data)
        done = self._get_done(exec_data)

        return self._get_obs(), reward, self._get_truncated(), done, self._get_info()

    def _move_up(self) -> tuple[str, float, bool, bool, dict[str, Any]]:
        if self._state.current_node.parent is None:
            return (
                self._get_obs(),
                -1,
                self._get_truncated(),
                self._get_done(),
                self._get_info(),
            )

        self._state.current_node = self._state.current_node.parent
        return (
            self._get_obs(),
            0,
            self._get_truncated(),
            self._get_done(),
            self._get_info(),
        )

    def _move_down(self) -> tuple[str, float, bool, bool, dict[str, Any]]:
        children = self._state.current_node.children()
        if children:
            self._state.current_node = random.choice(children)

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
        return replace(self.subtrees, self._state.current_node)

    def _add(self) -> Node:
        return add(self.subtrees, self._state.current_node)

    def _remove(self) -> Node:
        return remove(self._state.current_node)
