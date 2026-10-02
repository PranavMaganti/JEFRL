import logging
import random
from typing import Optional

import numpy as np
from rl.dqn import DQN
from rl.dqn import ReplayMemory
from rl.fuzzing_action import FuzzingAction
from rl.train import double_dqn_targets
import torch
from torch import optim
import torch.nn as nn
from transformer.ast_transformer import embed
from transformer.tokenizer import ASTTokenizer
from transformers import RobertaModel


NUM_TRAINING_STEPS = 50000  # Number of steps to train the agent for
LR = 1e-3  # Learning rate of the AdamW optimizer
REPLAY_MEMORY_SIZE = 10000  # Size of the replay buffer
MAX_MUTATIONS = 50  # Maximum number of mutations to apply to the code snippet
MAX_EPISODE_STEPS = 200  # Maximum number of actions (including moves) per episode

EPS_START = 1  # Starting value of epsilon
EPS_END = 0.05
EPS_DECAY = 0.99995  # Controls the rate of exponential decay of epsilon, higher means a slower decay
BATCH_SIZE = 32  # Number of transitions sampled from the replay buffer
GAMMA = 0.99  # Discount factor as mentioned in the previous section
TAU = 1  # Update rate of the target network
TARGET_UPDATE = 1000  # Number of steps after which the target network is updated
GRADIENT_CLIP = 1  # Maximum value of the gradient during backpropagation
WARM_UP_STEPS = 0  # Number of steps to collect transitions before training starts

TokenState = tuple[torch.Tensor, torch.Tensor]
# The next state of a transition: its tokens and valid-action mask, or None if the
# episode ended
NextState = Optional[tuple[TokenState, torch.Tensor]]


def get_state_embedding(
    states: list[TokenState],
    ast_net: RobertaModel,
    tokenizer: ASTTokenizer,
    device: torch.device,
) -> torch.Tensor:
    flattened_state = [item for pair in states for item in pair]
    batch = tokenizer.pad_batch(flattened_state, device)
    embeddings = embed(ast_net, batch["input_ids"], batch["attention_mask"])
    return embeddings.view(len(states), -1)


# Select action based on epsilon-greedy policy over the valid actions
def epsilon_greedy(
    ast_net: RobertaModel,
    tokenizer: ASTTokenizer,
    policy_net: DQN,
    state: TokenState,
    mask: np.ndarray,
    step: int,
    device: torch.device,
) -> int:
    eps_threshold = max(EPS_END, (EPS_DECAY**step) * EPS_START)

    if random.random() > eps_threshold:
        with torch.no_grad():
            values = policy_net(get_state_embedding([state], ast_net, tokenizer, device))
            values = values.masked_fill(
                ~torch.from_numpy(mask).to(device).unsqueeze(0), float("-inf")
            )
            logging.debug(f"Q-values: {values}")
            return int(values.argmax().item())

    logging.debug(f"Random action selected, epsilon = {eps_threshold}")
    return int(random.choice(np.flatnonzero(mask)))


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
    memory: ReplayMemory,
    device: torch.device,
    batch_size: int = BATCH_SIZE,
    gamma: float = GAMMA,
) -> float:
    if len(memory) < batch_size:
        return 0.0

    transitions = memory.sample(batch_size)
    states, actions, next_states, rewards = zip(*transitions)

    action_batch = torch.tensor(actions, dtype=torch.long, device=device)
    reward_batch = torch.tensor(rewards, dtype=torch.float32, device=device)
    dones = torch.tensor([s is None for s in next_states], device=device)

    state_action_values = (
        policy_net(get_state_embedding(list(states), ast_net, tokenizer, device))
        .gather(1, action_batch.unsqueeze(1))
        .squeeze(1)
    )

    # Terminal transitions still need a placeholder next state for batching; its
    # value is zeroed by the done flag
    next_tokens = [s[0] if s is not None else states[i] for i, s in enumerate(next_states)]
    all_actions = torch.ones(len(FuzzingAction), dtype=torch.bool)
    next_masks = torch.stack(
        [s[1] if s is not None else all_actions for s in next_states]
    ).to(device)

    with torch.no_grad():
        next_embeddings = get_state_embedding(next_tokens, ast_net, tokenizer, device)

    expected_state_action_values = double_dqn_targets(
        policy_net, target_net, reward_batch, next_embeddings, dones, next_masks, gamma
    )

    loss = nn.functional.smooth_l1_loss(state_action_values, expected_state_action_values)

    # Optimize both the DQN and the AST Transformer
    optimizer.zero_grad(set_to_none=True)
    loss.backward()

    torch.nn.utils.clip_grad_value_(policy_net.parameters(), GRADIENT_CLIP)  # type: ignore
    optimizer.step()

    return loss.item()
