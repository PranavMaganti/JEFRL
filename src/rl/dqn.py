import random
from collections import deque
from typing import NamedTuple, Optional

import numpy as np
import torch
import torch.nn.functional as F

Transition = NamedTuple(
    "Transition",
    [
        ("state", str),
        ("action", np.int64),
        ("next_state", Optional[str]),
        ("reward", float),
    ],
)

BatchTransition = NamedTuple(
    "BatchTransition",
    [
        ("states", tuple[str]),
        ("actions", tuple[np.int64]),
        ("next_states", tuple[Optional[str]]),
        ("rewards", tuple[float]),
    ],
)


class ReplayMemory:
    def __init__(self, capacity: int):
        self.memory: deque[Transition] = deque([], maxlen=capacity)

    def push(
        self,
        state: str,
        action: np.int64,
        next_state: Optional[str],
        reward: float,
    ):
        """Save a transition"""
        self.memory.append(Transition(state, action, next_state, reward))

    def sample(self, batch_size: int) -> list[Transition]:
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(torch.nn.Module):
    def __init__(self, n_observations: int, n_actions: int):
        super().__init__()  # type: ignore
        self.layer1 = torch.nn.Linear(n_observations, 728)
        self.layer2 = torch.nn.Linear(728, 728)
        self.layer3 = torch.nn.Linear(728, n_actions)

    def forward(self, x: torch.Tensor):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)
