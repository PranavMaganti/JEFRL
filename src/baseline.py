# Initial coverage: 14.7366%  Final coverage: 14.86%
from datetime import datetime
from itertools import count
import logging
import os
from pathlib import Path
import pickle
import sys
import traceback

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
save_folder_name = start.strftime("%Y-%m-%dT%H:%M:.%f")
INTERESTING_FOLDER = Path("corpus/interesting")
MAX_LEN = 512  # Maximum length of the AST fragment sequence

engine = V8Engine()
tokenizer = ASTTokenizer(vocab, token_to_id, MAX_LEN)


logging.info("Initialising environment")
env = FuzzingEnv(
    corpus,
    subtrees,
    engine,
    total_coverage,
    tokenizer,
    INTERESTING_FOLDER / save_folder_name,
)


logging.info("Initial coverage: %s", env.total_coverage.coverage())

episode_rewards: list[float] = []
execution_coverage: dict[tuple[int, int], float] = {}

try:
    while True:
        state, info = env.reset()
        t = 0
        episode_reward = 0
        for t in count():
            action = env.action_space.sample()
            next_state, reward, truncated, done, info = env.step(action)
            episode_reward += reward

            if env.total_executions % 100 == 0:
                execution_coverage[
                    (env.total_executions, env.total_actions)
                ] = env.total_coverage.coverage()

            if done or truncated:
                break

        episode_rewards.append(episode_reward)
        logging.info(f"Episode reward: {episode_reward}")

except Exception as e:
    traceback.print_exception(type(e), e, e.__traceback__)

finally:
    end = datetime.now()

    save_folder = Path("models") / save_folder_name
    os.makedirs(save_folder, exist_ok=True)

    final_coverage = env.total_coverage.coverage()
    total_steps = env.total_actions
    total_executions = env.total_executions

    with open(save_folder / "run_data.pkl", "wb") as f:
        pickle.dump(
            {
                "episode_rewards": episode_rewards,
                "execution_coverage": execution_coverage,
                "final_coverage": final_coverage,
                "total_steps": total_steps,
                "total_executions": total_executions,
            },
            f,
        )

    logging.info(
        f"Finished with final coverage: {env.total_coverage} in {end - start}",
    )
    logging.info(f"Average reward: {sum(episode_rewards) / len(episode_rewards)}")
    logging.info(f"Total steps: {env.total_actions}")
    logging.info(f"Total engine executions: {env.total_executions}")
