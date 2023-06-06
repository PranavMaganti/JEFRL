# Initial coverage: 14.73665% Final coverage: 14.78238%
from datetime import datetime
import logging
import os
from pathlib import Path
import pickle
import sys
import time
import traceback

from rl.dqn import DQN
from rl.dqn import ReplayMemory
from rl.env import FuzzingEnv
from rl.fuzzing_action import FuzzingAction
from rl.tokenizer import ASTTokenizer
from rl.train import epsilon_greedy
from rl.train import optimise_model
from rl.train import soft_update_params
import torch
from torch import optim
from transformers import get_linear_schedule_with_warmup
from transformers import RobertaConfig
from transformers import RobertaModel

from utils.js_engine import V8Engine
from utils.logging import setup_logging


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

LR = 1e-4  # Learning rate of the AdamW optimizer
NUM_EPISODES = 10000  # Number of episodes to train the agent for
MAX_LEN = 512  # Maximum length of the AST fragment sequence

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
    max_position_embeddings=MAX_LEN + 2,
)

# Load the ASTBERTa model
tokenizer = ASTTokenizer(vocab, token_to_id, MAX_LEN)
pretrained_model = torch.load("ASTBERTa/models/final/model_28000.pt")

if isinstance(pretrained_model, torch.nn.DataParallel):
    pretrained_model = pretrained_model.module

ast_net = RobertaModel.from_pretrained(
    pretrained_model_name_or_path=None,
    state_dict=pretrained_model.state_dict(),
    config=config,
).to(device)
# ast_net = torch.nn.DataParallel(ast_net, device_ids=[0, 1])
# ast_net = torch.load("ASTBERTa/models/final/model_28000.pt").to(device)

# Check types of the loaded model
assert isinstance(config, RobertaConfig)
# assert isinstance(ast_net, RobertaModel)


# Get number of actions from gym action space
n_actions = len(FuzzingAction)

# Number of observations is the size of the hidden state of the LSTM for both
# the target and context code
n_observations = hidden_size * 2

# Initialise policy and target networks
logging.info("Initialising policy and target networks")
policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

# policy_net = torch.nn.DataParallel(policy_net, device_ids=[0, 1])
# target_net = torch.nn.DataParallel(target_net, device_ids=[0, 1])

optimizer = optim.AdamW(
    [*ast_net.parameters(), *policy_net.parameters()],
    lr=LR,
    amsgrad=True,
)
lr_scheduler = get_linear_schedule_with_warmup(
    optimizer=optimizer, num_warmup_steps=2400, num_training_steps=50000
)
memory = ReplayMemory(10000)
update_count = 0

# Setup environment
start = datetime.now()
save_folder_name = start.strftime("%Y-%m-%dT%H:%M:.%f")

logging.info("Setting up environment")
INTERESTING_FOLDER = Path("corpus/interesting")

engine = V8Engine()
env = FuzzingEnv(
    corpus,
    subtrees,
    engine,
    total_coverage,
    tokenizer,
    INTERESTING_FOLDER / save_folder_name,
)

logging.info("Starting training")

total_steps = 0
episode_rewards: list[float] = []
execution_coverage: dict[tuple[int, int], float] = {}

try:
    for ep in range(NUM_EPISODES):
        state, info = env.reset()
        done, truncated = False, False
        episode_reward = 0

        while not done and not truncated:
            action = epsilon_greedy(
                policy_net, state, ast_net, tokenizer, env, total_steps, device
            )
            next_state, reward, truncated, done, info = env.step(action)
            total_steps += 1
            episode_reward += reward

            memory.push(state, action, next_state, reward)
            optimise_model(
                policy_net,
                target_net,
                ast_net,
                tokenizer,
                optimizer,
                lr_scheduler,
                memory,
                batch_size=32,
                device=device,
            )
            soft_update_params(policy_net, target_net)

            state = next_state
            update_count += 1

            if env.total_executions % 100 == 0:
                execution_coverage[
                    (env.total_executions, total_steps)
                ] = env.total_coverage.coverage()

        episode_rewards.append(episode_reward)

except Exception as e:
    traceback.print_exception(type(e), e, e.__traceback__)

finally:
    end = datetime.now()

    save_folder = Path("models") / save_folder_name
    os.makedirs(save_folder, exist_ok=True)

    torch.save(ast_net, save_folder / "ast_net.pt")
    torch.save(policy_net, save_folder / "policy_net.pt")
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
