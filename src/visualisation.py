import argparse
from collections import defaultdict
from cProfile import label
import json
import logging
from pathlib import Path
import pickle
import sys

import graphviz
import numpy as np
from optimum.bettertransformer import BetterTransformer
from preprocessing.normalise import normalize_ast
from preprocessing.sanitise import sanitise_ast
from rl.dqn import DQN
from rl.env import FuzzingEnv
from rl.fuzzing_action import FuzzingAction
from rl.train import epsilon_greedy
from rl.train import get_state_embedding
from rl.train import MAX_MUTATIONS
from rl.train import NUM_TRAINING_STEPS
import torch
from tqdm import tqdm
from traitlets import default
from transformer.ast_transformer import get_ast_transformer_model
from visualisation.ast_visualiser import visualise_ast

from utils.js_engine import Coverage
from utils.js_engine import Engine
from utils.loader import load_corpus
from utils.logging import setup_logging
from utils.seed import setup_seeds


parser = argparse.ArgumentParser()

parser.add_argument("--rl-path", type=Path, help="The path to the RL model folder")
parser.add_argument("--rl-step", type=int, help="The step of the RL model to load")
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
    "--visualise-corpus-dir",
    type=Path,
    default=Path("corpus/demo/"),
)
parser.add_argument(
    "--visualise-corpus-config",
    type=Path,
    default=Path("corpus/configs/default.json"),
)
parser.add_argument(
    "--output",
    type=Path,
    default=Path("data/visualisation/"),
    help="The path to the output directory where processed files will be stored",
)

args = parser.parse_args()


if not args.data_dir.exists():
    raise ValueError(f"Data directory {args.data_dir} does not exist")

if not args.rl_path.exists():
    raise ValueError(f"RL path {args.rl_path} does not exist")

rl_models_path = args.rl_path / "models"
rl_ast_net_path = rl_models_path / f"ast_net.pt"
rl_policy_net_path = rl_models_path / f"policy_net_{args.rl_step}.pt"

if not rl_ast_net_path.exists():
    raise ValueError(f"AST net model {rl_ast_net_path} does not exist")

if not rl_policy_net_path.exists():
    raise ValueError(f"Policy net model {rl_policy_net_path} does not exist")

if not args.engine_executable.exists():
    raise ValueError(f"Engine executable {args.engine_executable} does not exist")

if not args.visualise_corpus_dir.exists():
    raise ValueError(f"Corpus directory {args.visualise_corpus_dir} does not exist")

if not args.visualise_corpus_config.exists():
    raise ValueError(f"Corpus config {args.visualise_corpus_config} does not exist")

if not args.output.exists():
    args.output.mkdir(parents=True)

# System setup
sys.setrecursionlimit(10000)
setup_logging()
seed = setup_seeds()

engine = Engine.get_engine(args.engine_name, args.engine_executable)


# Load preprocessed data
logging.info("Loading preprocessed data")
with open(args.data_dir / "corpus.pkl", "rb") as f:
    data = pickle.load(f)

with open(args.data_dir / "vocab_data.pkl", "rb") as f:
    vocab_data = pickle.load(f)

with open(args.visualise_corpus_config, "r") as f:
    corpus_config = json.load(f)

subtrees = data["subtrees"]

vocab = vocab_data["vocab"]
token_to_id = vocab_data["token_to_id"]

# Load corpus
logging.info("Loading corpus")
corpus = load_corpus(engine, args.visualise_corpus_dir, corpus_config)

for state in tqdm(corpus, desc="Normalising corpus"):
    sanitise_ast(state.root)
    normalize_ast(state.root)

total_coverage = Coverage()
for state in tqdm(corpus, desc="Calculating coverage"):
    total_coverage |= state.exec_data.coverage

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
tokenizer, ast_net, config = get_ast_transformer_model(
    vocab, token_to_id, rl_ast_net_path, device
)
ast_net = BetterTransformer.transform(ast_net)
ast_net.eval()

# Get number of actions from gym action space
n_actions = len(FuzzingAction)
# Input size to the DQN is the size of the ASTBERTa hidden state * 2 (target and context)
n_observations = config.hidden_size * 2

policy_net = DQN(n_observations, n_actions).to(device)
policy_net.load_state_dict(torch.load(rl_policy_net_path, map_location=device))
policy_net.eval()


env = FuzzingEnv(
    corpus,
    subtrees,
    engine,
    total_coverage,
    tokenizer,
    max_mutations=MAX_MUTATIONS,
)


state, info = env.reset()
state_embedding = get_state_embedding(state, ast_net, tokenizer, device)


actions = []
rewards = []
done, truncated = False, False

graph = graphviz.Graph("pdf")
graph.attr(
    label=f"Recent actions: {actions[-5:]}, Recent rewards: {rewards[-5:]}, Is done: {done}",
    labelloc="t",
    fontsize="30",
)
visualise_ast(
    env._state.root,
    graph,
    defaultdict(int),
    env._state.target_node,
    env._state.context_node[-1],
)

graph.render("demo")

input("Press Enter to continue...")

while not done and not truncated:
    action = epsilon_greedy(
        policy_net, state_embedding, env, NUM_TRAINING_STEPS, device
    )
    next_state, reward, truncated, done, info = env.step(np.int64(action.item()))

    next_state_embedding = get_state_embedding(next_state, ast_net, tokenizer, device)

    state = next_state
    state_embedding = next_state_embedding

    actions.append(FuzzingAction(action.item()))
    rewards.append(reward)

    graph = graphviz.Graph("pdf")
    graph.attr(
        label=f"Recent actions: {actions[-5:]}, Recent rewards: {rewards[-5:]}, Is done: {done}",
        labelloc="t",
    )
    visualise_ast(
        env._state.root,
        graph,
        defaultdict(int),
        env._state.target_node,
        env._state.context_node[-1],
    )
    graph.render("demo")

    input("Press Enter to continue...")
