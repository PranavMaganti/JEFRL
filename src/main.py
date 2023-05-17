import logging
import sys
from itertools import count

import numpy as np
import torch
import tqdm
from torch import optim
from transformers import BatchEncoding, RobertaConfig, RobertaModel, RobertaTokenizer

from js_ast.analysis import scope_analysis
from rl.dqn import DQN, ReplayMemory
from rl.env import FuzzingAction, FuzzingEnv
from rl.train import epsilon_greedy, optimise_model, soft_update_params
from utils.js_engine import V8Engine
from utils.loader import get_subtrees, load_corpus
from utils.logging import setup_logging

# Logging setup
sys.setrecursionlimit(10000)
setup_logging()

# Environment setup
logging.info("Loading corpus")
engine = V8Engine()
corpus = load_corpus(engine)

logging.info("Initialising subtrees")
subtrees = get_subtrees(corpus)

logging.info("Analysing scopes")
for state in tqdm.tqdm(corpus):
    scope_analysis(state.current_node)

logging.info("Initialising environment")
env = FuzzingEnv(corpus, subtrees, engine)

LR = 1e-4  # Learning rate of the AdamW optimizer
NUM_EPISODES = 10000  # Number of episodes to train the agent for

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load CodeBERTa model
logging.info("Loading CodeBERTa model")

model_name = "huggingface/CodeBERTa-small-v1"
tokenizer: RobertaTokenizer = RobertaTokenizer.from_pretrained(model_name)  # type: ignore
config: RobertaConfig = RobertaConfig.from_pretrained(model_name)  # type: ignore
code_net: RobertaModel = RobertaModel.from_pretrained(model_name, config=config).to(device)  # type: ignore

# Check types of the loaded model
assert isinstance(tokenizer, RobertaTokenizer)
assert isinstance(config, RobertaConfig)
assert isinstance(code_net, RobertaModel)

# Get number of actions from gym action space
n_actions = len(FuzzingAction)
n_observations = config.hidden_size

# Initialise policy and target networks
logging.info("Initialising policy and target networks")
policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000)
update_count = 0

logging.info("Starting training")
for ep in range(NUM_EPISODES):
    state, info = env.reset()
    tokenized_state: BatchEncoding = tokenizer(
        state, return_tensors="pt", padding=True, truncation=True
    ).to(device)

    for t in count():
        action = epsilon_greedy(policy_net, code_net, tokenized_state, env, t, device)
        next_state, reward, truncated, done, info = env.step(np.int64(action.item()))

        memory.push(state, action, next_state, torch.Tensor([reward]))
        optimise_model(
            policy_net, target_net, code_net, tokenizer, optimizer, memory, device
        )
        soft_update_params(policy_net, target_net)

        if done or truncated:
            break

        state = next_state
        update_count += 1
