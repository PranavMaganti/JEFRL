# Initial coverage: 14.7366%  Final coverage: 14.86%
from datetime import datetime
import logging
import os
from pathlib import Path
import pickle
import sys
import traceback

import numpy as np
from rl.env import FuzzingEnv
from rl.tokenizer import ASTTokenizer
import torch
from rl.train import NUM_TRAINING_STEPS

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

fuzz_start = datetime.now()
save_folder_name = fuzz_start.strftime("%Y-%m-%dT%H:%M:.%f") + "_baseline"
data_save_folder = Path("data") / save_folder_name
os.makedirs(data_save_folder, exist_ok=True)

INTERESTING_FOLDER = Path("corpus/interesting")
MAX_FRAGMENT_SEQ_LEN = 512  # Maximum length of the AST fragment sequence

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
tokenizer = ASTTokenizer(vocab, token_to_id, MAX_FRAGMENT_SEQ_LEN, device)


logging.info("Initialising environment")
engine = V8Engine()
env = FuzzingEnv(
    corpus,
    subtrees,
    engine,
    total_coverage,
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
    while total_steps < NUM_TRAINING_STEPS:
        state, info = env.reset()
        done, truncated = False, False
        episode_reward: list[float] = []
        episode_action: list[tuple[int, str]] = []

        while not done and not truncated:
            action = env.action_space.sample()
            episode_action.append((action, env._state.target_node.type))

            start = datetime.now()
            next_state, reward, truncated, done, info = env.step(action)
            end = datetime.now()
            logging.info(f"Step time: {end - start}")

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
        f"Finished with final coverage: {env.total_coverage} in {end - fuzz_start}",
    )
    logging.info(
        f"Coverage increase: {env.total_coverage.coverage() - initial_coverage}"
    )
    logging.info(f"Average reward: {np.mean(episode_rewards_summed):.2f}")
    logging.info(f"Total steps: {env.total_actions}")
    logging.info(f"Total engine executions: {env.total_executions}")
