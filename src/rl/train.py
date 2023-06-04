import logging
import math
import random

import numpy as np
import torch
from torch import nn, optim
from transformers import RobertaModel

from rl.dqn import DQN, BatchTransition, ReplayMemory
from rl.env import FuzzingEnv
from rl.tokenizer import ASTTokenizer

EPS_START = 0.95  # Starting value of epsilon
EPS_END = 0.05
EPS_DECAY = 10000  # Controls the rate of exponential decay of epsilon, higher means a slower decay
BATCH_SIZE = 20  # Number of transitions sampled from the replay buffer
GAMMA = 0.9  # Discount factor as mentioned in the previous section
TAU = 0.005  # Update rate of the target network


# Select action based on epsilon-greedy policy
def epsilon_greedy(
    policy_net: DQN,
    state: tuple[torch.Tensor, torch.Tensor],
    ast_net: RobertaModel,
    tokenizer: ASTTokenizer,
    env: FuzzingEnv,
    step: int,
    device: torch.device,
) -> np.int64:
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1.0 * step / EPS_DECAY)
    if sample > eps_threshold:
        # Sample action from model depending of state
        with torch.no_grad():
            # Get the code snippet embedding
            state_embedding = get_state_embedding(
                [state], ast_net, tokenizer, device
            ).view(-1)
            return np.int64(policy_net(state_embedding).argmax().item())
    else:
        print(f"RANDOM ACTION: {eps_threshold}")
        # Sample random action
        return env.action_space.sample()


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


def get_state_embedding(
    state: list[tuple[torch.Tensor, torch.Tensor]],
    ast_net: RobertaModel,
    tokenizer: ASTTokenizer,
    device: torch.device,
) -> torch.Tensor:
    # Flatten state
    flattened_state = [item for state in state for item in state]
    # Get the code snippet embedding
    state_embedding = ast_net(
        **tokenizer.process_batch(flattened_state, device)
    ).pooler_output.view(len(state), -1)
    return state_embedding


def optimise_model(
    policy_net: DQN,
    target_net: DQN,
    ast_net: RobertaModel,
    ast_tokenizer: ASTTokenizer,
    optimizer: optim.Optimizer,
    memory: ReplayMemory,
    device: torch.device,
    batch_size: int = BATCH_SIZE,
    gamma: float = GAMMA,
):
    # If the replay buffer is not full, do not optimise
    if len(memory) < batch_size:
        return

    transitions = memory.sample(batch_size)
    states, actions, next_states, rewards = zip(*transitions)
    batch = BatchTransition(states, actions, next_states, rewards)  # type: ignore

    non_final_mask = torch.tensor(
        [s is not None for s in batch.next_states],
        device=device,
        dtype=torch.bool,
    )

    states_batch = get_state_embedding(batch.states, ast_net, ast_tokenizer, device)

    action_batch = torch.tensor(batch.actions).view(-1, 1).to(device)
    reward_batch = torch.tensor(batch.rewards).to(device)

    # Encode the state and get the Q values for the actions taken
    state_action_values = policy_net(states_batch).gather(1, action_batch)

    next_state_values = torch.zeros(batch_size, device=device)
    with torch.no_grad():
        non_final_next_states = [s for s in batch.next_states if s is not None]
        non_final_next_states = get_state_embedding(
            non_final_next_states, ast_net, ast_tokenizer, device
        )

        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]

    # Compute the expected Q values
    expected_state_action_values = (next_state_values * gamma) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    logging.info(f"Loss: {loss.item()}")

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()

    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)  # type: ignore
    optimizer.step()
