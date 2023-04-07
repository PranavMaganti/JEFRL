import math
import random
from itertools import count

import torch
from torch import nn, optim
from transformers import AutoTokenizer, RobertaConfig, RobertaModel

from environment.main import FuzzingEnv, ProgramState
from rl.dqn import DQN, ReplayMemory, Transition
from utils.js_engine import ExecutionData
from utils.loader import get_subtrees, load_corpus

CORPUS_PATH = "corpus"

corpus = load_corpus(CORPUS_PATH)
subtrees = get_subtrees(corpus)
seeds = [ProgramState(ast, ExecutionData()) for ast in corpus]
env = FuzzingEnv(5, 5, seeds, subtrees)

BATCH_SIZE = 16  # Number of transitions sampled from the replay buffer
GAMMA = 0.99  # Discount factor as mentioned in the previous section
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000  # Controls the rate of exponential decay of epsilon, higher means a slower decay
TAU = 0.005  # Update rate of the target network
LR = 1e-4  # Learning rate of the AdamW optimizer
NUM_EPISODES = 1000  # Number of episodes to train the agent for

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = "huggingface/CodeBERTa-small-v1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
config = RobertaConfig.from_pretrained(model_name)
code_model = RobertaModel.from_pretrained(model_name, config=config).to(device)


# Get number of actions from gym action space
n_actions = env.action_space.n
n_observations = config.hidden_size


policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())


optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000)


def epsilon_greedy(state, step: int):
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1.0 * step / EPS_DECAY)
    if sample > eps_threshold:
        with torch.no_grad():
            hidden_state = code_model(**state).last_hidden_state[0][0]
            return policy_net(hidden_state).argmax().view(1, 1)
    else:
        return torch.tensor(
            [[env.action_space.sample()]], device=device, dtype=torch.long
        )


def dict_cat(ds: list[dict]):
    return {k: torch.cat([d[k] for d in ds]) for k in ds[0].keys()}


def optimise_model(memory: ReplayMemory, device: torch.device):
    if len(memory) < BATCH_SIZE:
        return

    print("Optimising model")
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(
        tuple(map(lambda s: s is not None, batch.next_state)),
        device=device,
        dtype=torch.bool,
    )
    non_final_next_states = tokenizer(
        [s for s in batch.next_state if s is not None],
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(device)
    hidden_non_final_next_states = code_model(
        **non_final_next_states
    ).last_hidden_state[:, 0, :]

    state_batch = tokenizer(
        batch.state,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(device)
    action_batch = torch.cat(batch.action).to(device)
    reward_batch = torch.cat(batch.reward).to(device)

    hidden_state = code_model(**state_batch).last_hidden_state[:, 0, :]
    state_action_values = policy_net(hidden_state).gather(1, action_batch)

    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(
            hidden_non_final_next_states
        ).max(1)[0]
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()

    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()


update_count = 0

for ep in range(NUM_EPISODES):
    state, info = env.reset()
    encoded_state = tokenizer(
        state, return_tensors="pt", padding=True, truncation=True
    ).to(device)

    for t in count():
        action = epsilon_greedy(encoded_state, update_count)
        next_state, reward, done, info = env.step(action.item())
        if next_state is not None:
            break

        encoded_next_state = tokenizer(
            next_state,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(device)

        if done:
            next_state = None

        memory.push(state, action, next_state, torch.Tensor([reward]))

        optimise_model(memory, device)

        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[
                key
            ] * TAU + target_net_state_dict[key] * (1 - TAU)
        target_net.load_state_dict(target_net_state_dict)

        if done:
            break

        state = next_state
        update_count += 1
