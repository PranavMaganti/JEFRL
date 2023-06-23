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
from rl.tokenizer import ASTTokenizer
import torch
from torch import optim
from transformers import get_linear_schedule_with_warmup
from transformers import RobertaConfig
from transformers import RobertaModel

from utils.js_engine import V8Engine
from utils.logging import setup_logging


INTERESTING_FOLDER = Path("corpus/interesting")
ENGINE_VERSION = "8.5"

parser = argparse.ArgumentParser()
parser.add_argument(
    "--pretraining-path", type=str, help="The path to the pretraining model folder"
)
parser.add_argument(
    "--pretraining-step", type=int, help="The step of the pretraining model to load"
)

args = parser.parse_args()

pretrained_model = torch.load(
    f"out/pretraining/{args.pretraining_path}/model-{args.pretraining_step}.pt"
)

seed = random.randint(0, 2**32 - 1)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)


# System setup
sys.setrecursionlimit(10000)

# Logging setup
setup_logging()

# Load preprocessed data
logging.info("Loading preprocessed data")
with open(f"data/js-rl/corpus-{ENGINE_VERSION}.pkl", "rb") as f:
    data = pickle.load(f)

with open("ASTBERTa/vocab_data.pkl", "rb") as f:
    vocab_data = pickle.load(f)

corpus = data["corpus"]
subtrees = data["subtrees"]
total_coverage = data["total_coverage"]

vocab = vocab_data["vocab"]
token_to_id = vocab_data["token_to_id"]


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

vocab_size = len(vocab)  # size of vocabulary
intermediate_size = 2048  # embedding dimension
hidden_size = 512

num_hidden_layers = 3
num_attention_heads = 8
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

# Load the ASTBERTa model
tokenizer = ASTTokenizer(vocab, token_to_id, MAX_FRAGMENT_SEQ_LEN, device)


ast_net = RobertaModel.from_pretrained(
    pretrained_model_name_or_path=None,
    state_dict=pretrained_model.state_dict(),
    config=config,
).to(device)


assert isinstance(config, RobertaConfig)

# Initialise policy and target networks
logging.info("Initialising policy and target networks")

# Get number of actions from gym action space
n_actions = len(FuzzingAction)
# Input size to the DQN is the size of the ASTBERTa hidden state * 2 (target and context)
n_observations = hidden_size * 2

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
save_folder_name = fuzz_start.strftime("%Y-%m-%dT%H:%M:.%f")
data_save_folder = Path("data/") / save_folder_name
os.makedirs(data_save_folder, exist_ok=True)

logging.info("Setting up environment")
engine = V8Engine(version=ENGINE_VERSION)
env = FuzzingEnv(
    corpus,
    subtrees,
    engine,
    total_coverage,
    tokenizer,
    INTERESTING_FOLDER / save_folder_name,
    max_mutations=MAX_MUTATIONS,
)

# Save hyperparameters
with open(data_save_folder / "hyperparameters.json", "w") as f:
    f.write(
        json.dumps(
            {
                "num_training_steps": NUM_TRAINING_STEPS,
                "replay_memory_size": REPLAY_MEMORY_SIZE,
                "learning_rate": LR,
                "max_fragment_seq_len": MAX_FRAGMENT_SEQ_LEN,
                "max_mutations": MAX_MUTATIONS,
                "astberta_config": {
                    "vocab_size": vocab_size,
                    "intermediate_size": intermediate_size,
                    "hidden_size": hidden_size,
                    "num_hidden_layers": num_hidden_layers,
                    "num_attention_heads": num_attention_heads,
                    "dropout": dropout,
                },
                "eps_start": EPS_START,
                "eps_end": EPS_END,
                "eps_decay": EPS_DECAY,
                "gamma": GAMMA,
                "batch_size": BATCH_SIZE,
                "tau": TAU,
                "target_update": TARGET_UPDATE,
                "gradient_clip": GRADIENT_CLIP,
                # "grad_accumulation_steps": GRAD_ACCUMULATION_STEPS,
                "seed": seed,
                "engine_version": ENGINE_VERSION,
                "info": "Added new reward for decreased coverage, and added leaky_relu",
                "dqn arch": "512, 256",
            }
        )
    )


logging.info("Starting training")
total_steps = 0
initial_coverage = env.total_coverage.coverage()

episode_rewards: list[list[float]] = []
execution_coverage: dict[tuple[int, int], float] = {}
episode_coverage: list[float] = [initial_coverage]
episode_actions: list[list[tuple[int, str]]] = []

losses: list[float] = []


try:
    while total_steps < NUM_TRAINING_STEPS:
        state, info = env.reset()
        done, truncated = False, False
        episode_reward: list[float] = []
        episode_action: list[tuple[int, str]] = []

        while not done and not truncated:
            ep_start = datetime.now()
            action = epsilon_greedy(
                ast_net, tokenizer, policy_net, state, env, total_steps, device
            )
            episode_action.append((action.item(), env._state.target_node.type))

            start = datetime.now()
            next_state, reward, truncated, done, info = env.step(action.item())
            end = datetime.now()
            print(f"Step took {(end - start).total_seconds()} seconds")
            episode_reward.append(reward)
            total_steps += 1

            # next_state_embedding = get_state_embedding(next_state, ast_net, tokenizer)

            memory.push(
                state,
                action,
                next_state,
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
            if total_steps % 100 == 0:
                execution_coverage[
                    (env.total_executions, total_steps)
                ] = env.total_coverage.coverage()

            if total_steps % 1000 == 0:
                # torch.save(ast_net, data_save_folder / f"ast_net_{total_steps}.pt")

                current_coverage = env.total_coverage.coverage()
                total_executions = env.total_executions

                with open(data_save_folder / f"run_data_{total_steps}.pkl", "wb") as f:
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
                    data_save_folder / f"policy_net_{total_steps}.pt",
                )
                torch.save(
                    target_net.state_dict(),
                    data_save_folder / f"target_net_{total_steps}.pt",
                )
                torch.save(
                    ast_net.state_dict(), data_save_folder / f"ast_net_{total_steps}.pt"
                )

            ep_end = datetime.now()
            print(f"Episode took {(ep_end - ep_start).total_seconds()} seconds")

        episode_coverage.append(env.total_coverage.coverage())
        episode_rewards.append(episode_reward)
        episode_actions.append(episode_action)


except Exception as e:
    traceback.print_exception(type(e), e, e.__traceback__)

finally:
    end = datetime.now()
    episode_rewards_summed = [sum(episode) for episode in episode_rewards]

    logging.info(f"Initial coverage: {initial_coverage:.5%}")
    logging.info(
        f"Finished with final coverage: {env.total_coverage} in {end - fuzz_start}",
    )
    logging.info(
        f"Coverage increase: {(env.total_coverage.coverage() - initial_coverage):.5%}"
    )
    logging.info(f"Average reward: {np.mean(episode_rewards_summed):.2f}")
    logging.info(f"Total steps: {env.total_actions}")
    logging.info(f"Total engine executions: {env.total_executions}")
