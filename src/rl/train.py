import logging
import math
import random
from typing import Iterable

import numpy as np
from rl.dqn import BatchTransition
from rl.dqn import DQN
from rl.dqn import ReplayMemory
from rl.env import FuzzingEnv
import torch
from torch import nn
from torch import optim
from torch.nn.utils.rnn import pad_sequence
from transformers import BatchEncoding
from transformers import RobertaModel
from transformers import RobertaTokenizerFast


EPS_START = 0.95  # Starting value of epsilon
EPS_END = 0.05
EPS_DECAY = 1000  # Controls the rate of exponential decay of epsilon, higher means a slower decay
BATCH_SIZE = 16  # Number of transitions sampled from the replay buffer
GAMMA = 0.9  # Discount factor as mentioned in the previous section
TAU = 0.005  # Update rate of the target network


# Select action based on epsilon-greedy policy
def epsilon_greedy(
    policy_net: DQN,
    state: tuple[str, str],
    code_net: RobertaModel,
    code_tokenizer: RobertaTokenizerFast,
    code_lstm: nn.LSTM,
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
            state_embedding = target_context_code_embedding(
                [state], code_net, code_tokenizer, code_lstm, device
            )
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


# Split a tensor into multiple tensors based on a mapping with elements with
# the same mapping value being grouped together in the same tensor. Then append
# these into a batch with padding.
def split_tensor_by_mapping(tensor: torch.Tensor, mapping: list[int]) -> torch.Tensor:
    start = 0
    outputs: list[torch.Tensor] = []

    for i in range(1, len(tensor)):
        if mapping[i] == mapping[start]:
            continue

        outputs.append(tensor[start:i])
        start = i

    outputs.append(tensor[start:])

    return pad_sequence(outputs, batch_first=True)


# Embed a list of code snippets into a tensor. Handles long code sequences by
# splitting them into separate sequences and then passing them through an LSTM
def code_embedding(
    code: Iterable[str],
    code_net: RobertaModel,
    code_tokenizer: RobertaTokenizerFast,
    code_lstm: nn.LSTM,
    device: torch.device,
):
    # Tokenize the code, splitting long token sequences into multiple elements
    # of the batch
    tokens: BatchEncoding = code_tokenizer.batch_encode_plus(  # type: ignore
        list(code),
        max_length=512,
        return_tensors="pt",
        return_overflowing_tokens=True,
        padding=True,
        truncation=True,
    ).to(device)

    # Split inputs into several batches if necessary
    input_ids = torch.split(tokens["input_ids"], BATCH_SIZE)
    attention_mask = torch.split(tokens["attention_mask"], BATCH_SIZE)

    out = torch.Tensor().to(device)

    # Run the code through the RoBERTa model
    with torch.no_grad():
        for i in range(len(input_ids)):
            out = torch.cat(
                (
                    out,
                    code_net(
                        input_ids=input_ids[i], attention_mask=attention_mask[i]
                    ).pooler_output,
                )
            )

    overflow_mapping: list[int] = tokens["overflow_to_sample_mapping"]  # type: ignore

    # Split the output of the last hidden state into the different sequences
    sequence_outputs = split_tensor_by_mapping(out, overflow_mapping)

    # Run the LSTM on each sequence
    lstm_outputs, _ = code_lstm(sequence_outputs)

    # Return final output of the LSTM as the code embedding
    return lstm_outputs[:, -1, :]


def target_context_code_embedding(
    code: Iterable[tuple[str, str]],
    code_net: RobertaModel,
    code_tokenizer: RobertaTokenizerFast,
    code_lstm: nn.LSTM,
    device: torch.device,
):
    target_code, context_code = zip(*code)
    target_code_embedding = code_embedding(
        target_code, code_net, code_tokenizer, code_lstm, device
    )
    context_code_embedding = code_embedding(
        context_code, code_net, code_tokenizer, code_lstm, device
    )

    return torch.cat((target_code_embedding, context_code_embedding), dim=1)


def optimise_model(
    policy_net: DQN,
    target_net: DQN,
    code_net: RobertaModel,
    code_tokenizer: RobertaTokenizerFast,
    code_lstm: nn.LSTM,
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

    states_batch = target_context_code_embedding(
        batch.states, code_net, code_tokenizer, code_lstm, device
    )

    action_batch = torch.tensor(batch.actions).view(-1, 1).to(device)
    reward_batch = torch.tensor(batch.rewards).to(device)

    # Encode the state and get the Q values for the actions taken
    state_action_values = policy_net(states_batch).gather(1, action_batch)

    next_state_values = torch.zeros(batch_size, device=device)
    with torch.no_grad():
        non_final_next_states = target_context_code_embedding(
            [s for s in batch.next_states if s is not None],
            code_net,
            code_tokenizer,
            code_lstm,
            device,
        )
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]
        del non_final_mask
        del non_final_next_states

    # Compute the expected Q values
    expected_state_action_values = (next_state_values * gamma) + reward_batch
    del next_state_values

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
