# Initial coverage: 14.73665% Final coverage: 14.78238%

from itertools import count
import logging
import os
import sys
import time

from js_ast.analysis import scope_analysis
from rl.dqn import DQN
from rl.dqn import ReplayMemory
from rl.env import FuzzingAction
from rl.env import FuzzingEnv
from rl.train import epsilon_greedy
from rl.train import optimise_model
from rl.train import soft_update_params
import torch
from torch import optim
import tqdm
from transformers import RobertaConfig
from transformers import RobertaModel
from transformers import RobertaTokenizerFast

from utils.js_engine import V8Engine
from utils.loader import get_subtrees
from utils.loader import load_corpus
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

logging.info(f"Initial coverage {env.current_coverage}")

LR = 1e-3  # Learning rate of the AdamW optimizer
NUM_EPISODES = 10000  # Number of episodes to train the agent for

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load CodeBERTa model
logging.info("Loading CodeBERTa model")

model_name = "huggingface/CodeBERTa-small-v1"
tokenizer: RobertaTokenizerFast = RobertaTokenizerFast.from_pretrained(model_name)  # type: ignore
config: RobertaConfig = RobertaConfig.from_pretrained(model_name)  # type: ignore
code_net: RobertaModel = RobertaModel.from_pretrained(model_name, config=config).to(device)  # type: ignore

# Initialise LSTM
OUTPUT_DIM = 1024
code_lstm = torch.nn.LSTM(768, OUTPUT_DIM, 1, batch_first=True).to(device)


# Check types of the loaded model
assert isinstance(tokenizer, RobertaTokenizerFast)
assert isinstance(config, RobertaConfig)
assert isinstance(code_net, RobertaModel)

# Get number of actions from gym action space
n_actions = len(FuzzingAction)

# Number of observations is the size of the hidden state of the LSTM for both
# the target and context code
n_observations = OUTPUT_DIM * 2

# Initialise policy and target networks
logging.info("Initialising policy and target networks")
policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(
    [*code_lstm.parameters(), *policy_net.parameters()],
    lr=LR,
    amsgrad=True,
)
memory = ReplayMemory(10000)
update_count = 0

logging.info("Starting training")

total_time = 30 * 60  # Total time to run the agent for
start = time.time()

total_steps = 0
episode_rewards: list[float] = []

# for ep in range(NUM_EPISODES):
while time.time() - start < total_time:
    state, info = env.reset()
    done, truncated = False, False
    episode_reward = 0

    while not done and not truncated:
        action = epsilon_greedy(
            policy_net, state, code_net, tokenizer, code_lstm, env, total_steps, device
        )
        next_state, reward, truncated, done, info = env.step(action)
        total_steps += 1
        episode_reward += reward

        memory.push(state, action, next_state, reward)
        optimise_model(
            policy_net,
            target_net,
            code_net,
            tokenizer,
            code_lstm,
            optimizer,
            memory,
            device,
        )
        soft_update_params(policy_net, target_net)

        state = next_state
        update_count += 1

    episode_rewards.append(episode_reward)


logging.info(
    f"Finished with final coverage: {env.current_coverage}",
)
logging.info(f"Average reward: {sum(episode_rewards) / len(episode_rewards)}")
logging.info(f"Total steps: {env.total_actions}")
logging.info(f"Total engine executions: {env.total_executions}")
