import sys
from itertools import count

import torch
import tqdm
from torch import optim
from transformers import RobertaConfig, RobertaModel, RobertaTokenizer

from js_ast.analysis import scope_analysis
from rl.dqn import DQN, ReplayMemory
from rl.env import FuzzingEnv
from rl.train import epsilon_greedy, optimise_model, soft_update_params
from utils.js_engine import V8Engine
from utils.loader import get_subtrees, load_corpus

# sys.setrecursionlimit(10000)  # default is 1000 in my installation

# Environment setup
engine = V8Engine()
corpus = load_corpus(engine)
subtrees = get_subtrees(corpus)

for state in tqdm.tqdm(corpus):
    scope_analysis(state.current_node)

env = FuzzingEnv(corpus, subtrees, engine)


LR = 1e-4  # Learning rate of the AdamW optimizer
NUM_EPISODES = 10000  # Number of episodes to train the agent for


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = "huggingface/CodeBERTa-small-v1"
tokenizer = RobertaTokenizer.from_pretrained(model_name)
config = RobertaConfig.from_pretrained(model_name)
code_net = RobertaModel.from_pretrained(model_name, config=config).to(device)

# Get number of actions from gym action space
n_actions = env.action_space.n
n_observations = config.hidden_size

policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000)
update_count = 0

for ep in range(NUM_EPISODES):
    state, info = env.reset()
    tokenized_state = tokenizer(
        state, return_tensors="pt", padding=True, truncation=True
    ).to(device)

    for t in count():
        action = epsilon_greedy(policy_net, code_net, tokenized_state, env, t, device)
        next_state, reward, done, info = env.step(int(action.item()))

        memory.push(state, action, next_state, torch.Tensor([reward]))
        optimise_model(
            policy_net, target_net, code_net, tokenizer, optimizer, memory, device
        )
        soft_update_params(policy_net, target_net)

        if done:
            break

        state = next_state
        update_count += 1
