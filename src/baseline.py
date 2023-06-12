# Initial coverage: 14.7366%  Final coverage: 14.86%
import logging
import os
import pickle
import sys
import traceback
from datetime import datetime
from itertools import count
from pathlib import Path

import numpy as np
import torch
from optimum.bettertransformer import BetterTransformer
from transformers import RobertaConfig, RobertaModel

from rl.env import FuzzingEnv
from rl.tokenizer import ASTTokenizer
from utils.js_engine import V8Engine
from utils.logging import setup_logging

# System setup
sys.setrecursionlimit(10000)

# Logging setup
setup_logging()

logging.info("Loading preprocessed data")
with open("data/js-rl/corpus.pkl", "rb") as f:
    data = pickle.load(f)

with open("ASTBERTa/vocab_data.pkl", "rb") as f:
    vocab_data = pickle.load(f)

corpus = data["corpus"]
subtrees = data["subtrees"]
total_coverage = data["total_coverage"]

vocab = vocab_data["vocab"]
token_to_id = vocab_data["token_to_id"]

start = datetime.now()
save_folder_name = start.strftime("%Y-%m-%dT%H:%M:.%f") + "_baseline"
data_save_folder = Path("data") / save_folder_name
os.makedirs(data_save_folder, exist_ok=True)

INTERESTING_FOLDER = Path("corpus/interesting")
MAX_FRAGMENT_SEQ_LEN = 512  # Maximum length of the AST fragment sequence

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


vocab_size = len(vocab)  # size of vocabulary
intermediate_size = 3072  # embedding dimension
hidden_size = 768

num_hidden_layers = 6
num_attention_heads = 12
dropout = 0.1

config = RobertaConfig(
    vocab_size=vocab_size,
    hidden_size=hidden_size,
    num_hidden_layers=num_hidden_layers,
    num_attention_heads=num_attention_heads,
    intermediate_size=intermediate_size,
    hidden_dropout_prob=dropout,
    max_position_embeddings=MAX_FRAGMENT_SEQ_LEN + 2,
)

tokenizer = ASTTokenizer(vocab, token_to_id, MAX_FRAGMENT_SEQ_LEN, device)
pretrained_model = torch.load("ASTBERTa/models/final/model_28000.pt")

if isinstance(pretrained_model, torch.nn.DataParallel):
    pretrained_model = pretrained_model.module

ast_net = RobertaModel.from_pretrained(
    pretrained_model_name_or_path=None,
    state_dict=pretrained_model.state_dict(),
    config=config,
).to(device)
ast_net = BetterTransformer.transform(ast_net)


logging.info("Initialising environment")
engine = V8Engine()
env = FuzzingEnv(
    corpus,
    subtrees,
    engine,
    total_coverage,
    ast_net,
    tokenizer,
    INTERESTING_FOLDER / save_folder_name,
)


logging.info(f"Initial coverage: {env.total_coverage}")

initial_coverage = env.total_coverage.coverage()
total_steps = 0

episode_rewards: list[list[float]] = []
execution_coverage: dict[tuple[int, int], float] = {}
episode_coverage: list[float] = []
episode_actions: list[list[tuple[int, str]]] = []

try:
    while True:
        state, info = env.reset()
        done, truncated = False, False
        episode_reward: list[float] = []
        episode_action: list[tuple[int, str]] = []

        while not done and not truncated:
            action = env.action_space.sample()
            episode_action.append((action, env._state.target_node.type))

            next_state, reward, truncated, done, info = env.step(action)
            episode_reward.append(reward)

            total_steps += 1

            if total_steps % 100 == 0:
                execution_coverage[
                    (env.total_executions, env.total_actions)
                ] = env.total_coverage.coverage()

            if total_steps % 1000 == 0:
                current_coverage = env.total_coverage.coverage()
                total_executions = env.total_executions

                with open(data_save_folder / f"run_data_{total_steps}.pkl", "wb") as f:
                    pickle.dump(
                        {
                            "episode_actions": episode_actions,
                            "episode_rewards": episode_rewards,
                            "current_coverage": current_coverage,
                            "execution_coverage": execution_coverage,
                            "episode_coverage": episode_coverage,
                            "total_steps": total_steps,
                            "total_executions": total_executions,
                            "running_time": datetime.now() - start,
                        },
                        f,
                    )

        episode_rewards.append(episode_reward)
        episode_coverage.append(env.total_coverage.coverage())
        episode_actions.append(episode_action)
        logging.info(f"Episode reward: {sum(episode_reward)}")

except Exception as e:
    traceback.print_exception(type(e), e, e.__traceback__)

finally:
    end = datetime.now()
    final_coverage = env.total_coverage.coverage()
    episode_rewards_summed = [sum(episode) for episode in episode_rewards]

    logging.info(f"Initial coverage: {initial_coverage}")
    logging.info(
        f"Finished with final coverage: {env.total_coverage} in {end - start}",
    )
    logging.info(
        f"Coverage increase: {env.total_coverage.coverage() - initial_coverage}"
    )
    logging.info(f"Average reward: {np.mean(episode_rewards_summed):.2f}")
    logging.info(f"Total steps: {env.total_actions}")
    logging.info(f"Total engine executions: {env.total_executions}")
