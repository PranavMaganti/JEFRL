# Initial coverage: 14.7366%  Final coverage: 14.86%
import argparse
from datetime import datetime
import json
import logging
import os
from pathlib import Path
import pickle
import sys

from rl.env import FuzzingEnv
from rl.env import MAX_FRAGMENT_SEQ_LEN
from rl.train import NUM_TRAINING_STEPS
from transformer.tokenizer import ASTTokenizer

from utils.js_engine import Engine
from utils.logging import setup_logging
from utils.seed import setup_seeds


parser = argparse.ArgumentParser()

parser.add_argument(
    "--engine-name", type=str, default="v8", help="The engine to use for preprocessing"
)
parser.add_argument(
    "--engine-executable",
    type=Path,
    default="engines/v8/v8/out/fuzzbuild/d8",
    help="The path to the engine to use for preprocessing",
)
parser.add_argument(
    "--data-dir",
    type=Path,
    default=Path("data/"),
    help="The path to which the vocabulary and pretraining data was saved",
)
parser.add_argument(
    "--output",
    type=str,
    default=Path("out/baseline/"),
    help="The path to the output directory where processed files will be stored",
)

args = parser.parse_args()

if not args.data_dir.exists():
    raise ValueError(f"Data directory {args.data_dir} does not exist")

if not args.engine_executable.exists():
    raise ValueError(f"Engine executable {args.engine_executable} does not exist")

if not args.output.exists():
    os.makedirs(args.output, exist_ok=True)


# System setup
sys.setrecursionlimit(10000)
setup_logging()
seed = setup_seeds()

# Load preprocessed data
logging.info("Loading preprocessed data")
with open(args.data_dir / "corpus.pkl", "rb") as f:
    data = pickle.load(f)

with open(args.data_dir / "vocab_data.pkl", "rb") as f:
    vocab_data = pickle.load(f)


corpus = data["corpus"]
subtrees = data["subtrees"]
total_coverage = data["total_coverage"]

vocab = vocab_data["vocab"]
token_to_id = vocab_data["token_to_id"]

# Setup environment
fuzz_start = datetime.now()
data_folder_name = fuzz_start.strftime("%Y-%m-%dT%H:%M:.%f")
data_folder = args.output / data_folder_name
os.makedirs(data_folder, exist_ok=True)

interesting_tests_folder = data_folder / "interesting"
training_folder = data_folder / "training"
run_data_folder = training_folder / "run_data"

os.makedirs(interesting_tests_folder, exist_ok=True)
os.makedirs(training_folder, exist_ok=True)
os.makedirs(run_data_folder, exist_ok=True)

tokenizer = ASTTokenizer(vocab, token_to_id, MAX_FRAGMENT_SEQ_LEN)

logging.info("Initialising environment")
engine = Engine.get_engine(args.engine_name, args.engine_executable)
env = FuzzingEnv(
    corpus,
    subtrees,
    engine,
    total_coverage,
    tokenizer,
    interesting_tests_folder,
)

with open(data_folder / "hyperparameters.json", "w") as f:
    f.write(
        json.dumps(
            {
                "num_training_steps": NUM_TRAINING_STEPS,
                "seed": seed,
            }
        )
    )


logging.info(f"Initial coverage: {env.total_coverage}")

initial_coverage = env.total_coverage.coverage()
total_steps = 0

episode_rewards: list[list[float]] = []
execution_coverage: dict[tuple[int, int], float] = {}
episode_coverage: list[float] = []
episode_actions: list[list[tuple[int, str]]] = []

while total_steps < NUM_TRAINING_STEPS:
    state, info = env.reset()
    done, truncated = False, False
    episode_reward: list[float] = []
    episode_action: list[tuple[int, str]] = []

    while not done and not truncated:
        action = env.action_space.sample()
        episode_action.append((int(action), env._state.target_node.type))

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

            with open(run_data_folder / f"{total_steps}.pkl", "wb") as f:
                pickle.dump(
                    {
                        "episode_actions": episode_actions,
                        "episode_rewards": episode_rewards,
                        "episode_coverage": episode_coverage,
                        "execution_coverage": execution_coverage,
                        "current_coverage": current_coverage,
                        "total_steps": total_steps,
                        "total_executions": total_executions,
                        "failed_actions": env.failed_actions,
                        "running_time": datetime.now() - fuzz_start,
                        "action_times": env.action_times,
                        "exec_times": env.exec_times,
                        "code_gen_times": env.code_gen_times,
                    },
                    f,
                )

    episode_rewards.append(episode_reward)
    episode_coverage.append(env.total_coverage.coverage())
    episode_actions.append(episode_action)
    logging.info(f"Episode reward: {sum(episode_reward)}")
