# Initial coverage: 14.7366%  Final coverage: 14.86%
from itertools import count
import logging
import os
import sys
import time

from js_ast.analysis import scope_analysis
from rl.env import FuzzingEnv
import tqdm

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
corpus, coverage = load_corpus(engine)

logging.info("Initialising subtrees")
subtrees = get_subtrees(corpus)

logging.info("Analysing scopes")
for state in tqdm.tqdm(corpus):
    scope_analysis(state.target_node)

logging.info("Initialising environment")
env = FuzzingEnv(corpus, subtrees, 25, engine, coverage)

print(env.total_coverage.coverage() * 100)

# NUM_EPISODES = 10000  # Number of episodes to train the agent for

total_time = 120 * 60  # Total time to run the agent for
start = time.time()

episode_rewards: list[float] = []
try:
    while time.time() - start < total_time:
        state, info = env.reset()
        t = 0
        episode_reward = 0
        for t in count():
            action = env.action_space.sample()
            next_state, reward, truncated, done, info = env.step(action)
            episode_reward += reward

            if done or truncated:
                break

        episode_rewards.append(episode_reward)
        logging.info(f"Episode reward: {episode_reward}")

except KeyboardInterrupt:
    pass

logging.info(
    f"Finished baseline with final coverage: {env.total_coverage.coverage() * 100:.5f}%",
)
logging.info(f"Average episode reward: {sum(episode_rewards) / len(episode_rewards)}")
logging.info(f"Total steps: {env.total_actions}")
logging.info(f"Total engine executions: {env.total_executions}")
