from collections import deque
from collections import OrderedDict
import logging
from typing import Optional

import numpy as np
from rl.dqn import DQN
from rl.dqn import ReplayBuffer
import torch
from torch import optim
import torch.nn as nn
from transformer.ast_transformer import embed
from transformer.tokenizer import ASTTokenizer
from transformers import RobertaModel


NUM_TRAINING_STEPS = 800000  # Default number of steps to train the agent for
LR = 5e-4  # Learning rate of the AdamW optimizer
REPLAY_MEMORY_SIZE = 500000  # Size of the replay buffer (~3h of experience at ~45 steps/s)
MAX_MUTATIONS = 50  # Maximum number of mutations to perform on a code snippet
MAX_EPISODE_STEPS = 200  # Maximum number of actions (including moves) per episode

EPS_START = 1  # Starting value of epsilon
# Final epsilon. Higher than the usual 0.01-0.05 as the environment keeps changing
# (easy coverage runs out, the corpus grows), so the agent must keep exploring.
EPS_END = 0.1
# Steps over which epsilon decays linearly to EPS_END (~6h, or ~8% of a 72h run)
EPS_DECAY_STEPS = 1_000_000
STEPS_PER_UPDATE = 8  # Environment steps per DQN update (32 samples per transition)
N_STEP = 3  # Steps per return (Rainbow uses 3)
# Prioritised replay: alpha controls how strongly priorities skew sampling and beta
# the importance-sampling correction, annealed to 1 (the values from Schaul et al.)
PER_ALPHA = 0.6
PER_BETA_START = 0.4
PER_BETA_STEPS = 5_000_000
# Transitions per update. Large batches with fewer updates make much better use of a
# CPU learner (2.4x the samples per second of batch 64), as in Ape-X
BATCH_SIZE = 256
GAMMA = 0.98  # Discount factor as mentioned in the previous section
TAU = 1  # Update rate of the target network
TARGET_UPDATE = 1000  # Number of steps after which the target network is updated
GRADIENT_CLIP = 1  # Maximum value of the gradient during backpropagation
WARM_UP_STEPS = 50000  # Number of steps over which the learning rate warms up


class StateEmbedder:
    """Embeds (target, context) token sequences with the frozen AST Transformer.

    Embeddings are cached by token sequence: the context rarely changes between
    steps and the agent often revisits the same subtrees, so most lookups hit.
    """

    def __init__(
        self,
        ast_net: RobertaModel,
        tokenizer: ASTTokenizer,
        device: torch.device,
        cache_size: int = 50000,
        max_batch_size: int = 64,
    ):
        self.ast_net = ast_net
        self.tokenizer = tokenizer
        self.device = device
        self.cache_size = cache_size
        self.max_batch_size = max_batch_size
        self.cache: OrderedDict[tuple[int, ...], torch.Tensor] = OrderedDict()
        self.hits = 0
        self.misses = 0

    @torch.inference_mode()
    def _embed_sequences(self, seqs: list[tuple[int, ...]]):
        # Sort by length so each batch has little padding
        seqs = sorted(seqs, key=len)
        for i in range(0, len(seqs), self.max_batch_size):
            chunk = seqs[i : i + self.max_batch_size]
            batch = self.tokenizer.pad_batch(
                [torch.tensor(s, dtype=torch.long) for s in chunk], self.device
            )
            embeddings = embed(self.ast_net, batch["input_ids"], batch["attention_mask"])
            for seq, emb in zip(chunk, embeddings):
                self.cache[seq] = emb

    def __call__(self, states: list[tuple[list[int], list[int]]]) -> torch.Tensor:
        """Returns a [len(states), 2 * hidden] tensor of target and context embeddings."""
        keys = [(tuple(target), tuple(context)) for target, context in states]

        missing = {seq for pair in keys for seq in pair if seq not in self.cache}
        self.misses += len(missing)
        self.hits += 2 * len(keys) - len(missing)
        if missing:
            self._embed_sequences(list(missing))

        rows = []
        for target, context in keys:
            self.cache.move_to_end(target)
            self.cache.move_to_end(context)
            rows.append(torch.cat([self.cache[target], self.cache[context]]))

        while len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)

        return torch.stack(rows)


class NStepAccumulator:
    """Turns one worker's single-step transitions into n-step transitions:
    (s_t, a_t, r_t + γ r_t+1 + ... + γ^(n-1) r_t+n-1, s_t+n), so rewards reach the
    actions that led to them in fewer updates. At the end of an episode the
    remaining transitions are emitted with shorter returns and no bootstrapping.
    """

    def __init__(self, n: int, gamma: float):
        self.n = n
        self.gamma = gamma
        self.pending: deque[tuple[torch.Tensor, int, float]] = deque()

    def add(
        self,
        state: torch.Tensor,
        action: int,
        reward: float,
        next_state: torch.Tensor,
        done: bool,
        next_mask: np.ndarray,
    ) -> list[tuple]:
        """Adds a step, returning the transitions that are now complete as
        (state, action, return, next_state, done, next_mask, discount)."""
        self.pending.append((state, action, reward))
        ready = []

        if done:
            while self.pending:
                ready.append(self._pop(next_state, True, next_mask))
        elif len(self.pending) == self.n:
            ready.append(self._pop(next_state, False, next_mask))

        return ready

    def _pop(self, next_state, done, next_mask) -> tuple:
        state, action, _ = self.pending[0]
        ret = sum(r * self.gamma**k for k, (_, _, r) in enumerate(self.pending))
        discount = 0.0 if done else self.gamma ** len(self.pending)
        self.pending.popleft()
        return (state, action, ret, next_state, done, next_mask, discount)


def epsilon(step: int, end: float = EPS_END, decay_steps: int = EPS_DECAY_STEPS) -> float:
    """Linear decay from EPS_START to `end` over `decay_steps` steps."""
    progress = min(1.0, step / decay_steps) if decay_steps > 0 else 1.0
    return EPS_START + (end - EPS_START) * progress


@torch.no_grad()
def select_actions(
    policy_net: DQN,
    states: torch.Tensor,
    masks: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Epsilon-greedy over valid actions only. `masks` is a [N, A] bool tensor."""
    q_values = policy_net(states).masked_fill(~masks, float("-inf"))
    greedy = q_values.argmax(dim=1)
    random_actions = torch.multinomial(masks.float(), 1).squeeze(1)

    explore = torch.rand(states.shape[0], device=states.device) < eps
    logging.debug(f"Q-values: {q_values}")
    return torch.where(explore, random_actions, greedy)


def soft_update_params(policy_net: DQN, target_net: DQN, tau: float = TAU):
    # Soft update of the target network's weights
    # θ′ ← τ θ + (1 −τ )θ′
    target_net_state_dict = target_net.state_dict()
    policy_net_state_dict = policy_net.state_dict()
    for key in policy_net_state_dict:
        target_net_state_dict[key] = policy_net_state_dict[
            key
        ] * tau + target_net_state_dict[key] * (1 - tau)
    target_net.load_state_dict(target_net_state_dict)


def double_dqn_targets(
    policy_net: DQN,
    target_net: DQN,
    rewards: torch.Tensor,
    next_states: torch.Tensor,
    dones: torch.Tensor,
    next_masks: torch.Tensor,
    gamma: float | torch.Tensor,
) -> torch.Tensor:
    """r + γ Q_target(s', argmax_a' Q_policy(s', a')), with the argmax restricted to
    valid actions and no bootstrapping past the end of an episode. For n-step
    transitions r is the n-step return and γ is γ^n (per transition)."""
    with torch.no_grad():
        next_actions = (
            policy_net(next_states)
            .masked_fill(~next_masks, float("-inf"))
            .argmax(dim=1, keepdim=True)
        )
        next_values = target_net(next_states).gather(1, next_actions).squeeze(1)
        next_values = torch.where(dones, torch.zeros_like(next_values), next_values)
        return rewards + gamma * next_values


def optimise_model(
    policy_net: DQN,
    target_net: DQN,
    optimizer: optim.Optimizer,
    lr_scheduler: optim.lr_scheduler.LRScheduler,
    memory: ReplayBuffer,
    batch_size: int = BATCH_SIZE,
    beta: float = 1.0,
) -> Optional[torch.Tensor]:
    """Runs one Double DQN update on (n-step) transitions, updating their replay
    priorities. Returns the (detached) loss as a tensor rather than a float:
    calling .item() every update would make the CPU wait for the GPU, instead of
    stepping the environments while the update runs."""
    # Not enough transitions collected yet
    if len(memory) < batch_size:
        return None

    batch = memory.sample(batch_size, beta)

    state_action_values = (
        policy_net(batch.states).gather(1, batch.actions.unsqueeze(1)).squeeze(1)
    )
    expected_state_action_values = double_dqn_targets(
        policy_net,
        target_net,
        batch.rewards,
        batch.next_states,
        batch.dones,
        batch.next_masks,
        batch.discounts,
    )

    # Huber loss, weighted to correct for prioritised sampling
    td_errors = expected_state_action_values - state_action_values
    loss = (
        batch.weights
        * nn.functional.smooth_l1_loss(
            state_action_values, expected_state_action_values, reduction="none"
        )
    ).mean()
    memory.update_priorities(batch.indices, td_errors)

    # Optimize the model
    optimizer.zero_grad(set_to_none=True)
    loss.backward()

    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), GRADIENT_CLIP)  # type: ignore
    optimizer.step()
    lr_scheduler.step()

    return loss.detach()
