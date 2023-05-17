import math
import random

import torch
from torch import nn, optim
from transformers import BatchEncoding, RobertaModel, RobertaTokenizer

from rl.dqn import DQN, BatchTransition, ReplayMemory
from rl.env import FuzzingEnv

EPS_START = 0.95  # Starting value of epsilon
EPS_END = 0.05
EPS_DECAY = 10000  # Controls the rate of exponential decay of epsilon, higher means a slower decay
BATCH_SIZE = 10  # Number of transitions sampled from the replay buffer
GAMMA = 0.99  # Discount factor as mentioned in the previous section
TAU = 0.005  # Update rate of the target network


# Select action based on epsilon-greedy policy
def epsilon_greedy(
    policy_net: DQN,
    code_net: RobertaModel,
    state: BatchEncoding,
    env: FuzzingEnv,
    step: int,
    device: torch.device,
) -> torch.Tensor:
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1.0 * step / EPS_DECAY)
    if sample > eps_threshold:
        # Sample action from model depending of state
        with torch.no_grad():
            hidden_state = code_net(**state).last_hidden_state[0][0]
            return policy_net(hidden_state).argmax().view(1, 1)
    else:
        # Sample random action
        return torch.tensor(
            [[env.action_space.sample()]], device=device, dtype=torch.long
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
    code_net: RobertaModel,
    code_tokenizer: RobertaTokenizer,
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
        tuple(map(lambda s: s is not None, batch.next_states)),
        device=device,
        dtype=torch.bool,
    )
    non_final_next_states = code_tokenizer(
        [s for s in batch.next_states if s is not None],
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(device)
    hidden_non_final_next_states = code_net(**non_final_next_states).last_hidden_state[
        :, 0, :
    ]

    state_batch = code_tokenizer(
        batch.states,  # type: ignore
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(device)
    action_batch = torch.cat(batch.actions).to(device)
    reward_batch = torch.cat(batch.rewards).to(device)

    # Encode the state and get the Q values for the actions taken
    hidden_state = code_net(**state_batch).last_hidden_state[:, 0, :]
    state_action_values = policy_net(hidden_state).gather(1, action_batch)

    next_state_values = torch.zeros(batch_size, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(
            hidden_non_final_next_states
        ).max(1)[0]

    # Compute the expected Q values
    expected_state_action_values = (next_state_values * gamma) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()

    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)  # type: ignore
    optimizer.step()
