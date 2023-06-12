from datetime import datetime
import logging
import math
import random
import time

import numpy as np
from rl.dqn import BatchTransition
from rl.dqn import DQN
from rl.dqn import ReplayMemory
from rl.env import FuzzingEnv
from rl.tokenizer import ASTTokenizer
import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaModel


NUM_TRAINING_STEPS = 80000  # Number of episodes to train the agent for
LR = 5e-4  # Learning rate of the AdamW optimizer
REPLAY_MEMORY_SIZE = 8000  # Size of the replay buffer

EPS_START = 1  # Starting value of epsilon
EPS_END = 0.05
EPS_DECAY = 13000  # Controls the rate of exponential decay of epsilon, higher means a slower decay
BATCH_SIZE = 256  # Number of transitions sampled from the replay buffer
GAMMA = 0.95  # Discount factor as mentioned in the previous section
TAU = 0.005  # Update rate of the target network

# Weights for each action in epsilon-greedy policy to reduce probability of
# ending the episode early
ACTION_WEIGHTS = [1, 1, 1, 1, 1, 1, 1, 1, 0.5]


# Select action based on epsilon-greedy policy
def epsilon_greedy(
    policy_net: DQN,
    state: torch.Tensor,
    env: FuzzingEnv,
    step: int,
    device: torch.device,
) -> torch.Tensor:
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1.0 * step / EPS_DECAY)

    if sample > eps_threshold:
        # Sample action from model depending of state
        with torch.no_grad():
            # Get the code snippet embedding
            values = policy_net(state)
            print(f"VALUES: {values}")
            return values.argmax().view(1, 1)

    print(f"RANDOM ACTION: {eps_threshold}")
    # Sample random action
    return torch.tensor(
        [random.choices(range(env.action_space.n), ACTION_WEIGHTS)],
        device=device,
        dtype=torch.long,
    )


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


def optimise_model(
    policy_net: DQN,
    target_net: DQN,
    optimizer: optim.Optimizer,
    memory: ReplayMemory,
    device: torch.device,
    batch_size: int = BATCH_SIZE,
    gamma: float = GAMMA,
) -> float:
    # If the replay buffer is not full, do not optimise
    if len(memory) < batch_size:
        return 0.0

    transitions = memory.sample(batch_size)
    states, actions, next_states, rewards = zip(*transitions)
    batch = BatchTransition(states, actions, next_states, rewards)  # type: ignore

    non_final_mask = torch.tensor(
        tuple(map(lambda s: s is not None, batch.next_states)),
        device=device,
        dtype=torch.bool,
    )
    non_final_next_states = [s for s in batch.next_states if s is not None]

    states_batch = torch.cat(batch.states)
    next_states_batch = torch.cat(non_final_next_states)
    action_batch = torch.cat(batch.actions)
    reward_batch = torch.cat(batch.rewards)

    state_action_values = policy_net(states_batch).gather(1, action_batch)
    next_state_values = torch.zeros(batch_size, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(next_states_batch).max(1)[0]

    # Compute the expected Q values
    expected_state_action_values = (next_state_values * gamma) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad(set_to_none=True)
    loss.backward()

    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)  # type: ignore
    optimizer.step()

    return loss.item()
