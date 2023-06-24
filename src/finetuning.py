# Initial coverage: 14.73665% Final coverage: 14.78238%
import argparse
from datetime import datetime
import json
import logging
import os
from pathlib import Path
import pickle
import random
import sys
import traceback

import numpy as np
from rl.dqn import DQN
from rl.dqn import ReplayMemory
from rl.env import FuzzingEnv
from rl.env import MAX_FRAGMENT_SEQ_LEN

# from rl.train import GRAD_ACCUMULATION_STEPS
from rl.finetuning import BATCH_SIZE
from rl.finetuning import EPS_DECAY
from rl.finetuning import EPS_END
from rl.finetuning import EPS_START
from rl.finetuning import epsilon_greedy
from rl.finetuning import GAMMA
from rl.finetuning import GRADIENT_CLIP
from rl.finetuning import LR
from rl.finetuning import MAX_MUTATIONS
from rl.finetuning import NUM_TRAINING_STEPS
from rl.finetuning import optimise_model
from rl.finetuning import REPLAY_MEMORY_SIZE
from rl.finetuning import soft_update_params
from rl.finetuning import TARGET_UPDATE
from rl.finetuning import TAU
from rl.finetuning import WARM_UP_STEPS
from rl.fuzzing_action import FuzzingAction
import torch
from torch import optim
from transformer.ast_transformer import get_ast_transformer_model
from transformer.tokenizer import ASTTokenizer
from transformers import get_linear_schedule_with_warmup

from utils.js_engine import Engine
from utils.js_engine import V8Engine
from utils.logging import setup_logging
from utils.seed import setup_seeds


parser = argparse.ArgumentParser()

parser.add_argument(
    "--pretraining-path", type=Path, help="The path to the pretraining model folder"
)
parser.add_argument(
    "--pretraining-step", type=int, help="The step of the pretraining model to load"
)
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
    help="The path to which the vocabulary and pretraining data will be saved",
)
parser.add_argument(
    "--output",
    type=str,
    default=Path("out/finetuning/"),
    help="The path to the output directory where processed files will be stored",
)

args = parser.parse_args()


if not args.data_dir.exists():
    raise ValueError(f"Data directory {args.data_dir} does not exist")

if not args.pretraining_path.exists():
    raise ValueError(f"Pretraining path {args.pretraining_path} does not exist")

pretrained_model_path = args.pretraining_path / f"model_{args.pretraining_step}.pt"

if not pretrained_model_path.exists():
    raise ValueError(f"Pretraining model {pretrained_model_path} does not exist")

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


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
tokenizer, ast_net, config = get_ast_transformer_model(
    vocab, token_to_id, pretrained_model_path, device
)

# Initialise policy and target networks
logging.info("Initialising policy and target networks")

# Get number of actions from gym action space
n_actions = len(FuzzingAction)
# Input size to the DQN is the size of the ASTBERTa hidden state * 2 (target and context)
n_observations = config.hidden_size * 2

policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

for param in target_net.parameters():
    param.requires_grad = False

optimizer = optim.AdamW(
    [*ast_net.parameters(), *policy_net.parameters()],
    lr=LR,
    amsgrad=True,
)
lr_scheduler = get_linear_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=WARM_UP_STEPS,
    num_training_steps=NUM_TRAINING_STEPS,
)
memory = ReplayMemory(REPLAY_MEMORY_SIZE)

# Setup environment
fuzz_start = datetime.now()
data_folder_name = fuzz_start.strftime("%Y-%m-%dT%H:%M:.%f")
data_folder = args.output / data_folder_name
os.makedirs(data_folder, exist_ok=True)

interesting_tests_folder = data_folder / "interesting"
training_folder = data_folder / "training"
models_folder = training_folder / "models"
run_data_folder = data_folder / "run_data"

os.makedirs(interesting_tests_folder, exist_ok=True)
os.makedirs(training_folder, exist_ok=True)
os.makedirs(models_folder, exist_ok=True)
os.makedirs(run_data_folder, exist_ok=True)


logging.info("Setting up environment")
engine = Engine.get_engine(args.engine_name, args.engine_executable)
env = FuzzingEnv(
    corpus,
    subtrees,
    engine,
    total_coverage,
    tokenizer,
    interesting_tests_folder,
    max_mutations=MAX_MUTATIONS,
)

# Save hyperparameters
with open(data_folder / "hyperparameters.json", "w") as f:
    f.write(
        json.dumps(
            {
                "num_training_steps": NUM_TRAINING_STEPS,
                "replay_memory_size": REPLAY_MEMORY_SIZE,
                "learning_rate": LR,
                "max_fragment_seq_len": MAX_FRAGMENT_SEQ_LEN,
                "max_mutations": MAX_MUTATIONS,
                "eps_start": EPS_START,
                "eps_end": EPS_END,
                "eps_decay": EPS_DECAY,
                "gamma": GAMMA,
                "batch_size": BATCH_SIZE,
                "tau": TAU,
                "target_update": TARGET_UPDATE,
                "gradient_clip": GRADIENT_CLIP,
                "seed": seed,
                "pretrained_model_path": str(pretrained_model_path),
            }
        )
    )


logging.info("Starting training")
total_steps = 0
initial_coverage = env.total_coverage.coverage()

episode_rewards: list[list[float]] = []
execution_coverage: dict[int, float] = {}
action_coverage: dict[int, float] = {}

episode_coverage: list[float] = [initial_coverage]
episode_actions: list[list[tuple[int, str]]] = []

losses: list[float] = []


while total_steps < NUM_TRAINING_STEPS:
    state, info = env.reset()
    state_tensor = (
        torch.tensor(state[0], device=device),
        torch.tensor(state[1], device=device),
    )

    done, truncated = False, False
    episode_reward: list[float] = []
    episode_action: list[tuple[int, str]] = []

    while not done and not truncated:
        ep_start = datetime.now()
        action = epsilon_greedy(
            ast_net, tokenizer, policy_net, state_tensor, env, total_steps, device
        )
        episode_action.append((int(action.item()), env._state.target_node.type))

        start = datetime.now()
        next_state, reward, truncated, done, info = env.step(np.int64(action.item()))
        end = datetime.now()
        print(f"Step took {(end - start).total_seconds()} seconds")
        episode_reward.append(reward)
        total_steps += 1

        next_state_tensor = (
            torch.tensor(next_state[0], device=device),
            torch.tensor(next_state[1], device=device),
        )

        memory.push(
            state_tensor,
            action,
            next_state_tensor,
            torch.tensor([reward], device=device),
        )

        start = datetime.now()
        loss = optimise_model(
            ast_net,
            tokenizer,
            policy_net,
            target_net,
            optimizer,
            # lr_scheduler,
            memory,
            device,
        )
        end = datetime.now()
        print(f"Optimisation took {(end - start).total_seconds()} seconds")

        if total_steps % TARGET_UPDATE == 0:
            start = datetime.now()
            soft_update_params(policy_net, target_net)
            end = datetime.now()
            print(f"Soft update took {(end - start).total_seconds()} seconds")

        losses.append(loss)

        state = next_state
        state_tensor = next_state_tensor

        if total_steps % 100 == 0:
            execution_coverage[env.total_executions] = env.total_coverage.coverage()
            action_coverage[total_steps] = env.total_coverage.coverage()

        if total_steps % 1000 == 0:
            # torch.save(ast_net, data_save_folder / f"ast_net_{total_steps}.pt")

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
                        "losses": losses,
                    },
                    f,
                )
        if total_steps % 5000 == 0:
            torch.save(
                policy_net.state_dict(),
                models_folder / f"policy_net_{total_steps}.pt",
            )
            torch.save(
                target_net.state_dict(),
                models_folder / f"target_net_{total_steps}.pt",
            )
            torch.save(
                ast_net.state_dict(),
                models_folder / f"ast_net_{total_steps}.pt",
            )

        ep_end = datetime.now()
        print(f"Episode took {(ep_end - ep_start).total_seconds()} seconds")

    episode_coverage.append(env.total_coverage.coverage())
    episode_rewards.append(episode_reward)
    episode_actions.append(episode_action)
