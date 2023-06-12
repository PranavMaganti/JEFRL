# Initial coverage: 14.73665% Final coverage: 14.78238%
import json
import logging
import os
import pickle
import random
import sys
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from optimum.bettertransformer import BetterTransformer
from torch import optim
from transformers import RobertaConfig, RobertaModel

from rl.dqn import DQN, ReplayMemory
from rl.env import FuzzingEnv
from rl.fuzzing_action import FuzzingAction
from rl.tokenizer import ASTTokenizer
# from rl.train import GRAD_ACCUMULATION_STEPS
from rl.train import (ACTION_WEIGHTS, BATCH_SIZE, EPS_DECAY, EPS_END,
                      EPS_START, GAMMA, LR, NUM_TRAINING_STEPS,
                      REPLAY_MEMORY_SIZE, TAU, epsilon_greedy, optimise_model,
                      soft_update_params)
from utils.js_engine import V8Engine
from utils.logging import setup_logging

seed = 20
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)


INTERESTING_FOLDER = Path("corpus/interesting")
MAX_FRAGMENT_SEQ_LEN = 512  # Maximum length of the AST fragment sequence


# System setup
sys.setrecursionlimit(10000)

# Logging setup
setup_logging()

# Load preprocessed data
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


# Load the ASTBERTa model
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
# ast_net = torch.nn.DataParallel(ast_net, device_ids=[0, 1])
# ast_net = torch.load("ASTBERTa/models/final/model_28000.pt").to(device)

# Check types of the loaded model
assert isinstance(config, RobertaConfig)
# assert isinstance(ast_net, RobertaModel)

for param in ast_net.parameters():
    param.requires_grad = False


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
    [*policy_net.parameters()],
    lr=LR,
    amsgrad=True,
)
memory = ReplayMemory(REPLAY_MEMORY_SIZE)

# Setup environment
start = datetime.now()
save_folder_name = start.strftime("%Y-%m-%dT%H:%M:.%f")
data_save_folder = Path("data/") / save_folder_name
os.makedirs(data_save_folder, exist_ok=True)

# Save hyperparameters
with open(data_save_folder / "hyperparameters.json", "w") as f:
    f.write(
        json.dumps(
            {
                "num_training_steps": NUM_TRAINING_STEPS,
                "replay_memory_size": REPLAY_MEMORY_SIZE,
                "learning_rate": LR,
                "max_fragment_seq_len": MAX_FRAGMENT_SEQ_LEN,
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
                "action_weights": ACTION_WEIGHTS,
                # "grad_accumulation_steps": GRAD_ACCUMULATION_STEPS,
                "seed": seed,
            }
        )
    )


logging.info("Setting up environment")
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
            action = epsilon_greedy(policy_net, state, env, total_steps, device)
            episode_action.append((action.item(), env._state.target_node.type))

            start = datetime.now()
            next_state, reward, truncated, done, info = env.step(action.item())
            end = datetime.now()
            print(f"Step took {(end - start).total_seconds()} seconds")

            episode_reward.append(reward)
            total_steps += 1

            memory.push(
                state,
                action,
                next_state,
                torch.tensor([reward], device=device),
            )

            start = datetime.now()
            loss = optimise_model(
                policy_net,
                target_net,
                optimizer,
                memory,
                device,
            )
            end = datetime.now()
            print(f"Optimisation took {(end - start).total_seconds()} seconds")

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
                torch.save(
                    policy_net, data_save_folder / f"policy_net_{total_steps}.pt"
                )
                torch.save(
                    target_net, data_save_folder / f"target_net_{total_steps}.pt"
                )
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
                            "running_time": datetime.now() - start,
                            "losses": losses,
                        },
                        f,
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
        f"Finished with final coverage: {env.total_coverage} in {end - start}",
    )
    logging.info(
        f"Coverage increase: {(env.total_coverage.coverage() - initial_coverage):.5%}"
    )
    logging.info(f"Average reward: {np.mean(episode_rewards_summed):.2f}")
    logging.info(f"Total steps: {env.total_actions}")
    logging.info(f"Total engine executions: {env.total_executions}")
