import argparse
from datetime import datetime
import json
import logging
import os
from pathlib import Path
import pickle
import sys

from rl.dqn import DQN
from rl.dqn import ReplayMemory
from rl.env import FuzzingEnv
from rl.env import MAX_FRAGMENT_SEQ_LEN
from rl.env import SharedFuzzingState
from rl.finetuning import BATCH_SIZE
from rl.finetuning import EPS_DECAY
from rl.finetuning import EPS_END
from rl.finetuning import EPS_START
from rl.finetuning import epsilon_greedy
from rl.finetuning import GAMMA
from rl.finetuning import GRADIENT_CLIP
from rl.finetuning import LR
from rl.finetuning import MAX_EPISODE_STEPS
from rl.finetuning import MAX_MUTATIONS
from rl.finetuning import NUM_TRAINING_STEPS
from rl.finetuning import optimise_model
from rl.finetuning import REPLAY_MEMORY_SIZE
from rl.finetuning import soft_update_params
from rl.finetuning import TARGET_UPDATE
from rl.finetuning import TAU
from rl.fuzzing_action import FuzzingAction
import torch
from torch import optim
from transformer.ast_transformer import get_ast_transformer_model

from utils.device import get_device
from utils.js_engine import Engine
from utils.js_engine import TIMEOUT
from utils.logging import setup_logging
from utils.seed import setup_seeds


parser = argparse.ArgumentParser()

parser.add_argument(
    "--pretraining-path", type=Path, help="The path to the pretraining model folder"
)
parser.add_argument(
    "--pretraining-step",
    type=str,
    default="best",
    help="The step of the pretraining model to load (or 'best')",
)
parser.add_argument(
    "--engine-name", type=str, default="v8", help="The engine to fuzz"
)
parser.add_argument(
    "--engine-executable",
    type=Path,
    default="engines/v8/v8/out/fuzzbuild/d8",
    help="The path to the engine executable",
)
parser.add_argument(
    "--engine-mode", choices=["auto", "reprl", "subprocess", "remote"], default="auto"
)
parser.add_argument("--engine-arg", action="append", default=[])
parser.add_argument("--timeout", type=float, default=TIMEOUT)
parser.add_argument("--num-steps", type=int, default=NUM_TRAINING_STEPS)
parser.add_argument(
    "--data-dir",
    type=Path,
    default=Path("data/"),
    help="The path to which the vocabulary and pretraining data was saved",
)
parser.add_argument(
    "--output",
    type=Path,
    default=Path("out/finetuning/"),
    help="The path to the output directory where processed files will be stored",
)

parser.add_argument(
    "--engine-address",
    default="localhost:9000",
    help="host:port of an engine server (src/utils/engine_server.py) for --engine-mode remote",
)
args = parser.parse_args()


if not args.data_dir.exists():
    raise ValueError(f"Data directory {args.data_dir} does not exist")

if args.pretraining_path is None or not args.pretraining_path.exists():
    raise ValueError(f"Pretraining path {args.pretraining_path} does not exist")

pretrained_model_path = args.pretraining_path / f"model_{args.pretraining_step}.pt"

if not pretrained_model_path.exists():
    raise ValueError(f"Pretraining model {pretrained_model_path} does not exist")

if args.engine_mode != "remote" and not args.engine_executable.exists():
    raise ValueError(f"Engine executable {args.engine_executable} does not exist")

os.makedirs(args.output, exist_ok=True)

# Setup output folders
fuzz_start = datetime.now()
data_folder = args.output / fuzz_start.strftime("%Y-%m-%dT%H-%M-%S")
interesting_tests_folder = data_folder / "interesting"
training_folder = data_folder / "training"
models_folder = training_folder / "models"
run_data_folder = training_folder / "run_data"
for folder in (interesting_tests_folder, models_folder, run_data_folder):
    os.makedirs(folder, exist_ok=True)

# System setup
sys.setrecursionlimit(10000)
setup_logging(data_folder / "finetuning.log")
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


device = get_device()
logging.info(f"Using device {device}")
tokenizer, ast_net, config = get_ast_transformer_model(
    vocab, token_to_id, pretrained_model_path, device
)

# Initialise policy and target networks
logging.info("Initialising policy and target networks")

# Get number of actions from gym action space
n_actions = len(FuzzingAction)
# Input size to the DQN is the size of the AST Transformer hidden state * 2 (target and context)
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
memory = ReplayMemory(REPLAY_MEMORY_SIZE)


logging.info("Setting up environment")
engine = Engine.get_engine(
    args.engine_name, args.engine_executable, args.engine_mode, args.timeout, args.engine_arg,
    address=args.engine_address,
)
shared = SharedFuzzingState(corpus, total_coverage)
env = FuzzingEnv(
    shared,
    subtrees,
    engine,
    tokenizer,
    interesting_tests_folder,
    max_mutations=MAX_MUTATIONS,
    max_steps=MAX_EPISODE_STEPS,
)

# Save hyperparameters
with open(data_folder / "hyperparameters.json", "w") as f:
    json.dump(
        {
            "num_training_steps": args.num_steps,
            "replay_memory_size": REPLAY_MEMORY_SIZE,
            "learning_rate": LR,
            "max_fragment_seq_len": MAX_FRAGMENT_SEQ_LEN,
            "max_mutations": MAX_MUTATIONS,
            "max_episode_steps": MAX_EPISODE_STEPS,
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
            "engine": type(engine).__name__,
            "device": str(device),
        },
        f,
        indent=2,
    )


def to_tensors(state: tuple[list[int], list[int]]) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.tensor(state[0], dtype=torch.long),
        torch.tensor(state[1], dtype=torch.long),
    )


logging.info("Starting training")
total_steps = 0
initial_coverage = env.total_coverage.coverage()

episode_rewards: list[list[float]] = []
execution_coverage: dict[int, float] = {}
action_coverage: dict[int, float] = {}

episode_coverage: list[float] = [initial_coverage]
losses: list[float] = []

try:
    while total_steps < args.num_steps:
        state, info = env.reset()
        mask = info["action_mask"]
        state_tensor = to_tensors(state)

        done = False
        episode_reward: list[float] = []

        while not done:
            action = epsilon_greedy(
                ast_net, tokenizer, policy_net, state_tensor, mask, total_steps, device
            )

            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward.append(reward)
            total_steps += 1

            next_mask = info["action_mask"]
            next_state_tensor = to_tensors(next_state)

            memory.push(
                state_tensor,
                action,
                None if done else (next_state_tensor, torch.from_numpy(next_mask)),
                reward,
            )

            loss = optimise_model(
                ast_net,
                tokenizer,
                policy_net,
                target_net,
                optimizer,
                memory,
                device,
            )
            losses.append(loss)

            if total_steps % TARGET_UPDATE == 0:
                soft_update_params(policy_net, target_net)

            state_tensor, mask = next_state_tensor, next_mask

            if total_steps % 100 == 0:
                execution_coverage[env.total_executions] = env.total_coverage.coverage()
                action_coverage[total_steps] = env.total_coverage.coverage()

            if total_steps % 1000 == 0:
                logging.info(
                    f"steps={total_steps} execs={env.total_executions} coverage={env.total_coverage}"
                )
                with open(run_data_folder / f"{total_steps}.pkl", "wb") as f:
                    pickle.dump(
                        {
                            "episode_rewards": episode_rewards,
                            "episode_coverage": episode_coverage,
                            "execution_coverage": execution_coverage,
                            "action_coverage": action_coverage,
                            "execution_errors": dict(env.exec_errors),
                            "actions": dict(env.action_counts),
                            "current_coverage": env.total_coverage.coverage(),
                            "total_steps": total_steps,
                            "total_executions": env.total_executions,
                            "failed_actions": env.num_failed_actions,
                            "running_time": datetime.now() - fuzz_start,
                            "losses": losses[-10000:],
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

        episode_coverage.append(env.total_coverage.coverage())
        episode_rewards.append(episode_reward)
finally:
    torch.save(ast_net.state_dict(), models_folder / "ast_net_final.pt")
    engine.close()
