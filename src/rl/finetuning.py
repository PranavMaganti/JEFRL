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
import torch
from torch import optim
import torch.nn as nn
from transformer.tokenizer import ASTTokenizer
from transformers import RobertaModel


NUM_TRAINING_STEPS = 50000  # Number of episodes to train the agent for
LR = 1e-3  # Learning rate of the AdamW optimizer
REPLAY_MEMORY_SIZE = 10000  # Size of the replay buffer
MAX_MUTATIONS = 50  # Maximum number of mutations to apply to the code snippet

EPS_START = 1  # Starting value of epsilon
EPS_END = 0.05
EPS_DECAY = 0.99995  # Controls the rate of exponential decay of epsilon, higher means a slower decay
BATCH_SIZE = 32  # Number of transitions sampled from the replay buffer
GAMMA = 0.99  # Discount factor as mentioned in the previous section
TAU = 1  # Update rate of the target network
TARGET_UPDATE = 1000  # Number of steps after which the target network is updated
GRADIENT_CLIP = 1  # Maximum value of the gradient during backpropagation
WARM_UP_STEPS = 0  # Number of steps to collect transitions before training starts

# Weights for each action in epsilon-greedy policy to reduce probability of
# ending the episode early


def get_state_embedding(
    state: list[tuple[torch.Tensor, torch.Tensor]],
    ast_net: RobertaModel,
    tokenizer: ASTTokenizer,
) -> torch.Tensor:
    flattened_state = [item for sublist in state for item in sublist]
    batch = tokenizer.pad_batch(flattened_state)
    state_embedding = ast_net(**batch).pooler_output.view(len(state), -1)
    return state_embedding


# Select action based on epsilon-greedy policy
def epsilon_greedy(
    ast_net: RobertaModel,
    tokenizer: ASTTokenizer,
    policy_net: DQN,
    state: tuple[torch.Tensor, torch.Tensor],
    env: FuzzingEnv,
    step: int,
    device: torch.device,
) -> torch.Tensor:
    sample = random.random()
    eps_threshold = max(EPS_END, (EPS_DECAY**step) * EPS_START)

    if sample > eps_threshold:
        # Sample action from model depending of state
        with torch.no_grad():
            # Get the code snippet embedding
            values = policy_net(get_state_embedding([state], ast_net, tokenizer))
            logging.debug(f"Q-values: {values}")
            return values.argmax().view(1, 1)

    logging.debug(f"Random action selected, epsilon = {eps_threshold}")
    # Sample random action
    return torch.tensor(
        [env.action_space.sample()],
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
    ast_net: RobertaModel,
    tokenizer: ASTTokenizer,
    policy_net: DQN,
    target_net: DQN,
    optimizer: optim.Optimizer,
    # lr_scheduler: optim.lr_scheduler.LambdaLR,
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

    # states_batch = torch.cat(batch.states)
    # next_states_batch = torch.cat(non_final_next_states)
    action_batch = torch.cat(batch.actions)
    reward_batch = torch.cat(batch.rewards)

    states_batch_embedded = get_state_embedding(batch.states, ast_net, tokenizer)

    state_action_values = policy_net(states_batch_embedded).gather(1, action_batch)
    next_state_values = torch.zeros(batch_size, device=device)
    with torch.no_grad():
        next_states_batch_embedded = get_state_embedding(
            non_final_next_states, ast_net, tokenizer
        )
        next_state_values[non_final_mask] = (
            target_net(next_states_batch_embedded)
            .gather(1, policy_net(next_states_batch_embedded).argmax(1, keepdim=True))
            .squeeze()
        )

    # Compute the expected Q values
    expected_state_action_values = (next_state_values * gamma) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad(set_to_none=True)
    loss.backward()

    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), GRADIENT_CLIP)  # type: ignore
    optimizer.step()
    # lr_scheduler.step()

    return loss.item()
