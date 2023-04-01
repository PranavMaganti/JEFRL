import collections

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from environment.fuzzing_action import FuzzingAction
from utils.js_engine import ExecutionData
from nodes.main import Node
from utils.mutation import replace


class FuzzingAction(Enum):
    REPLACE = 0
    ADD = 1
    REMOVE = 2
    MOVE_UP = 3
    MOVE_DOWN = 4


class ProgramState:
    def __init__(self, program: Node, execution_data: ExecutionData):
        self.program = program
        self.current_node = program

        self.crash = execution_data.return_code != 0
        self.coverage = execution_data.hit_edges / execution_data.num_edges

    def __repr__(self):
        return f"ProgramState(crash={self.crash}, coverage={self.coverage})"

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
        self.action_space = spaces.Discrete(action_dim)
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

    def _get_obs(self):
        return {"state": self._state}

    def _get_info(self):
        return {}

    def _get_reward(self, execution_data: ExecutionData):
        if execution_data.return_code == 0:
            return

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Choose the agent's location uniformly at random
        self._state = self.np_random.choice(self.seeds)

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, action: int):
        fuzzing_action = self._id_to_action[action]
        match fuzzing_action:
            case FuzzingAction.MOVE_UP:
                if self._state.current_node.parent is not None:
                    self._state.current_node = self._state.current_node.parent

                return self._get_obs(), 0, False, self._get_info()

            case FuzzingAction.MOVE_DOWN:
                children = self._state.current_node.children()
                if children:
                    self._state.current_node = np.random.choice(children)

                return self._get_obs(), 0, False, self._get_info()

            case FuzzingAction.REPLACE:
                new_node = replace(self.subtrees, self._state.current_node)
                if new_node is not self._state.current_node:
                    self._state.current_node = new_node
