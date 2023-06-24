from collections import deque
import random
from typing import Generic, NamedTuple, Optional, TypeVar

import numpy as np
import torch
import torch.nn.functional as F


T = TypeVar("T")


class Transition(NamedTuple, Generic[T]):
    state: T
    action: torch.Tensor
    next_state: Optional[T]
    reward: torch.Tensor


class BatchTransition(NamedTuple, Generic[T]):
    states: tuple[T]
    actions: tuple[torch.Tensor]
    next_states: tuple[Optional[T]]
    rewards: tuple[torch.Tensor]


class ReplayMemory(Generic[T]):
    def __init__(self, capacity: int):
        self.memory: deque[Transition] = deque([], maxlen=capacity)

    def push(
        self,
        state: T,
        action: torch.Tensor,
        next_state: Optional[T],
        reward: torch.Tensor,
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
        x = F.leaky_relu(self.layer1(x))
        x = F.leaky_relu(self.layer2(x))
        x = F.leaky_relu(self.layer3(x))
        return self.layer4(x)
