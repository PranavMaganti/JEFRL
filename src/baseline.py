import logging
import os
import random
import sys
from itertools import count

import torch
import tqdm
from torch import optim
from transformers import RobertaConfig, RobertaModel, RobertaTokenizerFast

from js_ast.analysis import scope_analysis
from rl.dqn import DQN, ReplayMemory
from rl.env import FuzzingAction, FuzzingEnv
from rl.train import epsilon_greedy, optimise_model, soft_update_params
from utils.js_engine import V8Engine
from utils.loader import get_subtrees, load_corpus
from utils.logging import setup_logging

# System setup
os.environ["TOKENIZERS_PARALLELISM"] = "true"
sys.setrecursionlimit(10000)

# Logging setup
setup_logging()

# Environment setup
logging.info("Loading corpus")
engine = V8Engine()
corpus = load_corpus(engine)

logging.info("Initialising subtrees")
subtrees = get_subtrees(corpus)

logging.info("Analysing scopes")
for state in tqdm.tqdm(corpus):
    scope_analysis(state.target_node)

logging.info("Initialising environment")
env = FuzzingEnv(corpus, subtrees, 25, engine)

NUM_EPISODES = 10000  # Number of episodes to train the agent for

for ep in range(NUM_EPISODES):
    state, info = env.reset()
    t = 0
    for t in count():
        action = env.action_space.sample()
        next_state, reward, truncated, done, info = env.step(action)

        if done or truncated:
            break
