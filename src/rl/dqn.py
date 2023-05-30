from collections import deque
import random
from typing import NamedTuple, Optional

import numpy as np
import torch
import torch.nn.functional as F


Transition = NamedTuple(
    "Transition",
    [
        ("state", tuple[torch.Tensor, torch.Tensor]),
        ("action", np.int64),
        ("next_state", Optional[tuple[torch.Tensor, torch.Tensor]]),
        ("reward", float),
    ],
)

BatchTransition = NamedTuple(
    "BatchTransition",
    [
        ("states", tuple[tuple[torch.Tensor, torch.Tensor]]),
        ("actions", tuple[np.int64]),
        ("next_states", tuple[Optional[tuple[torch.Tensor, torch.Tensor]]]),
        ("rewards", tuple[float]),
    ],
)


class ReplayMemory:
    def __init__(self, capacity: int):
        self.memory: deque[Transition] = deque([], maxlen=capacity)

    def push(
        self,
        state: tuple[torch.Tensor, torch.Tensor],
        action: np.int64,
        next_state: Optional[tuple[torch.Tensor, torch.Tensor]],
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
        self.layer1 = torch.nn.Linear(n_observations, 2048)
        self.layer2 = torch.nn.Linear(2048, 1024)
        self.layer3 = torch.nn.Linear(1024, 512)
        self.layer4 = torch.nn.Linear(512, n_actions)

    def forward(self, x: torch.Tensor):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        return self.layer4(x)
