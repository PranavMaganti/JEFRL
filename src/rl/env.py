import random
from enum import IntEnum
from typing import Optional

import gymnasium as gym
from gymnasium import spaces

from js_ast.nodes import Node
from js_ast import escodegen
from utils.js_engine import CoverageData, Engine, ExecutionData
from utils.mutation import add, remove, replace


class FuzzingAction(IntEnum):
    REPLACE = 0
    ADD = 1
    REMOVE = 2
    MOVE_UP = 3
    MOVE_DOWN = 4
    END = 5


class ProgramState:
    def __init__(self, program: Node):
        self.program = program
        self.current_node = program

    def get_current_code(self) -> Optional[str]:
        try:
            return escodegen.generate(self.current_node)  # type: ignore
        except Exception:
            print("Error generating code")
            return None

    def __str__(self):
        return self.__repr__()


class FuzzingEnv(gym.Env):
    metadata = {}

    def __init__(
        self,
        corpus: list[ProgramState],
        subtrees: dict[str, list[Node]],
        engine: Engine,
        render_mode=None,
    ):
        self.action_space: spaces.Discrete = spaces.Discrete(len(FuzzingAction))
        self.render_mode = render_mode

        self.corpus = corpus
        self.subtrees = subtrees
        self.engine = engine

        self._state: ProgramState
        self.steps_since_increased_coverage = 0
        self.current_coverage = CoverageData()

    def _get_obs(self) -> str:
        obs = self._state.get_current_code()
        return obs if obs else ""

    def _get_info(self):
        return {}

    def _get_reward(self, exec_data: ExecutionData):
        if exec_data.is_crash() != 0:
            self.steps_since_increased_coverage = 0
            return 10

        new_coverage = exec_data.coverage_data | self.current_coverage

        # new coverage is the same as the current coverage
        if new_coverage == self.current_coverage:
            return 0

        # new coverage has increased total coverage
        print(f"New coverage - Total: {new_coverage.coverage()}")
        self.current_coverage = new_coverage
        self.steps_since_increased_coverage = 0

        return 5

    def _get_done(self, exec_data: Optional[ExecutionData] = None):
        return (
            exec_data and exec_data.is_crash()
        ) or self.steps_since_increased_coverage > 500

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Choose the agent's location uniformly at random
        self._state = random.choice(self.corpus)
        self.steps_since_increased_coverage = 0

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, action: int):
        print(
            f"Steps since increased coverage: {self.steps_since_increased_coverage}, action: {action}"
        )
        self.steps_since_increased_coverage += 1
        new_node = self._state.current_node

        match action:
            case FuzzingAction.MOVE_UP:
                if self._state.current_node.parent is not None:
                    self._state.current_node = self._state.current_node.parent

                return self._get_obs(), 0, self._get_done(), self._get_info()

            case FuzzingAction.MOVE_DOWN:
                children = self._state.current_node.children()
                if children:
                    self._state.current_node = random.choice(children)

                return self._get_obs(), 0, self._get_done(), self._get_info()

            case FuzzingAction.END:
                return self._get_obs(), 0, True, self._get_info()
            case FuzzingAction.REPLACE:
                new_node = replace(self.subtrees, self._state.current_node)
            case FuzzingAction.ADD:
                new_node = add(self.subtrees, self._state.current_node)
            case FuzzingAction.REMOVE:
                new_node = remove(self._state.current_node)

        if new_node is self._state.current_node:
            return self._get_obs(), 0, self._get_done(), self._get_info()

        self._state.current_node = new_node
        exec_data = self.engine.execute(self._state.program)
        if not exec_data:
            return self._get_obs(), 0, True, self._get_info()

        reward = self._get_reward(exec_data)
        done = self._get_done(exec_data)

        return self._get_obs(), reward, done, self._get_info()
