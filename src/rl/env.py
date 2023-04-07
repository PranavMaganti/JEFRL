import random
from enum import Enum
from typing import Optional

import gymnasium as gym
from gymnasium import spaces

from ast.nodes import Node
from ast import escodegen
from utils.js_engine import ExecutionData, execute_test
from utils.mutation import add, remove, replace


class FuzzingAction(Enum):
    REPLACE = 0
    ADD = 1
    REMOVE = 2
    MOVE_UP = 3
    MOVE_DOWN = 4


class ProgramState:
    def __init__(self, program: Node, exec_data: ExecutionData):
        self.program = program
        self.current_node = program
        self.exec_data = exec_data

    def get_current_code(self) -> Optional[str]:
        try:
            return escodegen.generate(self.current_node)  # type: ignore
        except Exception:
            print("Error generating code")
            return None

    def __repr__(self):
        return f"ProgramState(crash={self.exec_data.is_crash()}, coverage={self.exec_data.coverage()})"

    def __str__(self):
        return self.__repr__()


class FuzzingEnv(gym.Env):
    metadata = {}

    def __init__(
        self,
        action_dim: int,
        observation_dim: int,
        seeds: list[ProgramState],
        subtrees: dict[str, list[Node]],
        render_mode=None,
    ):
        self.action_space: spaces.Discrete = spaces.Discrete(action_dim)
        self.observation_space = spaces.Discrete(observation_dim)
        self.render_mode = render_mode

        self.seeds = seeds
        self.subtrees = subtrees
        self._state: ProgramState

        self._id_to_action = {
            0: FuzzingAction.REPLACE,
            1: FuzzingAction.ADD,
            2: FuzzingAction.REMOVE,
            3: FuzzingAction.MOVE_UP,
            4: FuzzingAction.MOVE_DOWN,
        }

        self.steps_since_increased_coverage = 0

    def _get_obs(self) -> Optional[str]:
        return self._state.get_current_code()

    def _get_info(self):
        return {}

    def _get_reward(self, exec_data: ExecutionData):
        if exec_data.is_crash() != 0:
            return 10

        if exec_data.coverage() > self._state.exec_data.coverage():
            return 5
        elif exec_data.coverage() < self._state.exec_data.coverage():
            return -1

        return 0

    def _get_done(self, exec_data: Optional[ExecutionData] = None):
        if not exec_data:
            self.steps_since_increased_coverage += 1
            return self.steps_since_increased_coverage > 500

        if exec_data.coverage() > self._state.exec_data.coverage():
            self.steps_since_increased_coverage = 0
        else:
            self.steps_since_increased_coverage += 1

        return exec_data.is_crash() or self.steps_since_increased_coverage > 200

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Choose the agent's location uniformly at random
        self._state = self.np_random.choice(self.seeds)

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, action: int):
        print(f"Steps since increased coverage: {self.steps_since_increased_coverage}")
        fuzzing_action = self._id_to_action[action]
        new_node = self._state.current_node

        match fuzzing_action:
            case FuzzingAction.MOVE_UP:
                if self._state.current_node.parent is not None:
                    self._state.current_node = self._state.current_node.parent

                return self._get_obs(), 0, self._get_done(), self._get_info()

            case FuzzingAction.MOVE_DOWN:
                children = self._state.current_node.children()
                if children:
                    self._state.current_node = random.choice(children)

                return self._get_obs(), 0, self._get_done(), self._get_info()

            case FuzzingAction.REPLACE:
                new_node = replace(self.subtrees, self._state.current_node)
            case FuzzingAction.ADD:
                new_node = add(self.subtrees, self._state.current_node)
            case FuzzingAction.REMOVE:
                new_node = remove(self._state.current_node)

        if new_node is self._state.current_node:
            return self._get_obs(), 0, self._get_done(), self._get_info()

        self._state.current_node = new_node
        exec_data = execute_test(self._state.program)
        if not exec_data:
            return self._get_obs(), 0, True, self._get_info()

        reward = self._get_reward(exec_data) if exec_data else 0
        done = self._get_done(exec_data)
        self._state.exec_data = exec_data

        return self._get_obs(), reward, done, self._get_info()
