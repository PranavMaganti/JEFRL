from collections import deque
import random
from typing import Generic, NamedTuple, Optional, TypeVar

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
    """Replay memory of arbitrary objects. Used when fine-tuning the AST
    Transformer, where token sequences have to be re-embedded on every update. A
    next_state of None marks the end of an episode."""

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


class ReplayBatch(NamedTuple):
    states: torch.Tensor  # [B, D]
    actions: torch.Tensor  # [B]
    rewards: torch.Tensor  # [B] (n-step) return
    next_states: torch.Tensor  # [B, D] state n steps later
    dones: torch.Tensor  # [B] episode ended within the n steps
    next_masks: torch.Tensor  # [B, A] valid actions in next_state
    discounts: torch.Tensor  # [B] gamma^n for the bootstrapped value
    indices: torch.Tensor  # [B] buffer positions, to update priorities
    weights: torch.Tensor  # [B] importance-sampling weights (1 without priorities)


class ReplayBuffer:
    """Fixed-size ring buffer of transitions between (frozen) state embeddings,
    stored in preallocated tensors so sampling is a single indexing operation.

    With alpha > 0 it is a prioritised replay buffer (Schaul et al. 2016):
    transitions are sampled in proportion to priority^alpha, where the priority is
    the magnitude of their last TD error, so rare surprising transitions (coverage
    increases, crashes) are replayed more often. Importance-sampling weights
    (annealed by beta) correct the bias this introduces.
    """

    def __init__(
        self,
        capacity: int,
        state_dim: int,
        n_actions: int,
        device: torch.device,
        dtype: torch.dtype = torch.float16,
        alpha: float = 0.0,
        priority_eps: float = 1e-3,
    ):
        self.capacity = capacity
        self.device = device
        self.alpha = alpha
        self.priority_eps = priority_eps
        self.states = torch.zeros(capacity, state_dim, dtype=dtype, device=device)
        self.next_states = torch.zeros(capacity, state_dim, dtype=dtype, device=device)
        self.actions = torch.zeros(capacity, dtype=torch.long, device=device)
        self.rewards = torch.zeros(capacity, dtype=torch.float32, device=device)
        self.dones = torch.zeros(capacity, dtype=torch.bool, device=device)
        self.next_masks = torch.zeros(
            capacity, n_actions, dtype=torch.bool, device=device
        )
        self.discounts = torch.zeros(capacity, dtype=torch.float32, device=device)
        # priority^alpha of each transition
        self.priorities = torch.zeros(capacity, dtype=torch.float32, device=device)
        # Kept on the device so tracking it never waits for the GPU
        self.max_priority = torch.tensor(1.0, device=device)
        self.pos = 0
        self.size = 0

    def push(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
        next_masks: torch.Tensor,
        discounts: torch.Tensor,
    ):
        """Adds a batch of transitions."""
        n = states.shape[0]
        idx = (torch.arange(n, device=self.device) + self.pos) % self.capacity

        self.states[idx] = states.to(self.device, self.states.dtype)
        self.next_states[idx] = next_states.to(self.device, self.states.dtype)
        self.actions[idx] = actions.to(self.device)
        self.rewards[idx] = rewards.to(self.device, torch.float32)
        self.dones[idx] = dones.to(self.device)
        self.next_masks[idx] = next_masks.to(self.device)
        self.discounts[idx] = discounts.to(self.device, torch.float32)
        # New transitions get the highest priority so they're replayed at least once
        self.priorities[idx] = self.max_priority**self.alpha

        self.pos = (self.pos + n) % self.capacity
        self.size = min(self.size + n, self.capacity)

    def sample(self, batch_size: int, beta: float = 1.0) -> ReplayBatch:
        if self.alpha > 0:
            probs = self.priorities[: self.size]
            idx = torch.multinomial(probs, batch_size, replacement=True)
            # w_i = (N * P(i))^-beta, normalised so the largest weight is 1
            p = probs[idx] / probs.sum()
            weights = (self.size * p) ** (-beta)
            weights = weights / weights.max()
        else:
            idx = torch.randint(0, self.size, (batch_size,), device=self.device)
            weights = torch.ones(batch_size, device=self.device)

        return ReplayBatch(
            self.states[idx].float(),
            self.actions[idx],
            self.rewards[idx],
            self.next_states[idx].float(),
            self.dones[idx],
            self.next_masks[idx],
            self.discounts[idx],
            idx,
            weights,
        )

    def update_priorities(self, indices: torch.Tensor, td_errors: torch.Tensor):
        if self.alpha <= 0:
            return
        priorities = td_errors.detach().abs().float() + self.priority_eps
        self.priorities[indices] = priorities**self.alpha
        self.max_priority = torch.maximum(self.max_priority, priorities.max())

    def __len__(self):
        return self.size

    _TENSORS = [
        "states",
        "next_states",
        "actions",
        "rewards",
        "dones",
        "next_masks",
        "discounts",
        "priorities",
    ]

    def state_dict(self) -> dict:
        # Only the filled part. Moving it to the CPU already copies it (on a GPU),
        # so no extra clone: the buffer can be several GB
        state = {
            name: getattr(self, name)[: self.size].detach().cpu()
            for name in self._TENSORS
        }
        return {
            **state,
            "pos": self.pos,
            "size": self.size,
            "max_priority": float(self.max_priority),
            "alpha": self.alpha,
        }

    def load_state_dict(self, state: dict):
        size = min(state["size"], self.capacity)
        for name in self._TENSORS:
            getattr(self, name)[:size] = state[name][:size].to(self.device)
        self.size = size
        self.pos = state["pos"] % self.capacity if size == self.capacity else size
        self.max_priority = torch.tensor(state["max_priority"], device=self.device)
        if state["alpha"] != self.alpha:
            # Stored as priority^alpha, so reset rather than mix exponents
            self.priorities[:size] = self.max_priority**self.alpha


class DQN(torch.nn.Module):
    """Q-network over (target, context) state embeddings. With dueling=True it
    estimates the state's value and each action's advantage separately
    (Wang et al. 2016), which helps when many actions have similar values, as
    most moves do here."""

    def __init__(self, n_observations: int, n_actions: int, dueling: bool = True):
        super().__init__()  # type: ignore
        self.dueling = dueling
        self.layer1 = torch.nn.Linear(n_observations, 2048)
        self.layer2 = torch.nn.Linear(2048, 1024)
        self.layer3 = torch.nn.Linear(1024, 512)
        self.layer4 = torch.nn.Linear(512, 256)
        self.layer5 = torch.nn.Linear(256, n_actions)
        if dueling:
            self.value4 = torch.nn.Linear(512, 256)
            self.value5 = torch.nn.Linear(256, 1)

    def forward(self, x: torch.Tensor):
        x = F.leaky_relu(self.layer1(x))
        x = F.leaky_relu(self.layer2(x))
        x = F.leaky_relu(self.layer3(x))
        advantage = self.layer5(F.leaky_relu(self.layer4(x)))
        if not self.dueling:
            return advantage

        value = self.value5(F.leaky_relu(self.value4(x)))
        return value + advantage - advantage.mean(dim=1, keepdim=True)
