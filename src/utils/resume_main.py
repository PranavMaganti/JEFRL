# # Initial coverage: 14.73665% Final coverage: 14.78238%
# from datetime import datetime
# import json
# import logging
# import os
# from pathlib import Path
# import pickle
# import sys
# import traceback

# import numpy as np
# from rl.dqn import DQN
# from rl.dqn import ReplayMemory
# from rl.env import FuzzingEnv
# from rl.fuzzing_action import FuzzingAction
# from rl.tokenizer import ASTTokenizer
# from rl.train import ACTION_WEIGHTS
# from rl.train import BATCH_SIZE
# from rl.train import EPS_DECAY
# from rl.train import EPS_END
# from rl.train import EPS_START
# from rl.train import epsilon_greedy
# from rl.train import GAMMA
# from rl.train import GRAD_ACCUMULATION_STEPS
# from rl.train import LR
# from rl.train import NUM_TRAINING_STEPS
# from rl.train import optimise_model
# from rl.train import REPLAY_MEMORY_SIZE
# from rl.train import soft_update_params
# from rl.train import TAU
# import torch
# from torch import optim
# from transformers import RobertaConfig
# from transformers import RobertaModel

# from utils.js_engine import V8Engine
# from utils.loader import load_program_states
# from utils.logging import setup_logging


# INTERESTING_FOLDER = Path("corpus/interesting")
# MAX_FRAGMENT_SEQ_LEN = 512  # Maximum length of the AST fragment sequence

# RESUME_FOLDER = "2023-06-11T18:59:.557829"
# RESUME_STEP = 9000

# # System setup
# sys.setrecursionlimit(10000)

# # Logging setup
# setup_logging()

# # Load preprocessed data
# logging.info("Loading preprocessed data")
# with open("data/js-rl/corpus.pkl", "rb") as f:
#     data = pickle.load(f)

# with open("ASTBERTa/vocab_data.pkl", "rb") as f:
#     vocab_data = pickle.load(f)

# with open(f"data/{RESUME_FOLDER}/run_data_{RESUME_STEP}.pkl", "rb") as f:
#     resume_data = pickle.load(f)

# resume_corpus, resume_coverage = load_program_states(INTERESTING_FOLDER / RESUME_FOLDER)

# print(resume_coverage)
# print(data["total_coverage"])

# corpus = data["corpus"] + resume_corpus
# subtrees = data["subtrees"]
# total_coverage = data["total_coverage"] | resume_coverage

# vocab = vocab_data["vocab"]
# token_to_id = vocab_data["token_to_id"]

# episode_actions = resume_data["episode_actions"]
# episode_rewards = resume_data["episode_rewards"]
# episode_coverage = resume_data["episode_coverage"]
# execution_coverage = resume_data["execution_coverage"]
# current_coverage = resume_data["current_coverage"]
# total_steps = resume_data["total_steps"]
# total_executions = resume_data["total_executions"]
# losses = resume_data["losses"]

# print(total_coverage )


# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# vocab_size = len(vocab)  # size of vocabulary
# intermediate_size = 3072  # embedding dimension
# hidden_size = 768

# num_hidden_layers = 6
# num_attention_heads = 12
# dropout = 0.1

# config = RobertaConfig(
#     vocab_size=vocab_size,
#     hidden_size=hidden_size,
#     num_hidden_layers=num_hidden_layers,
#     num_attention_heads=num_attention_heads,
#     intermediate_size=intermediate_size,
#     hidden_dropout_prob=dropout,
#     max_position_embeddings=MAX_FRAGMENT_SEQ_LEN + 2,
# )


# # Load the ASTBERTa model
# tokenizer = ASTTokenizer(vocab, token_to_id, MAX_FRAGMENT_SEQ_LEN)
# pretrained_model = torch.load(
#     Path("data") / RESUME_FOLDER / f"ast_net_{RESUME_STEP}.pt"
# )

# if isinstance(pretrained_model, torch.nn.DataParallel):
#     pretrained_model = pretrained_model.module

# ast_net = RobertaModel.from_pretrained(
#     pretrained_model_name_or_path=None,
#     state_dict=pretrained_model.state_dict(),
#     config=config,
# ).to(device)
# # ast_net = torch.nn.DataParallel(ast_net, device_ids=[0, 1])
# # ast_net = torch.load("ASTBERTa/models/final/model_28000.pt").to(device)

# # Check types of the loaded model
# assert isinstance(config, RobertaConfig)
# # assert isinstance(ast_net, RobertaModel)

# # Initialise policy and target networks
# logging.info("Initialising policy and target networks")

# # Get number of actions from gym action space
# n_actions = len(FuzzingAction)
# # Input size to the DQN is the size of the ASTBERTa hidden state * 2 (target and context)
# n_observations = hidden_size * 2

# policy_net = torch.load(f"data/{RESUME_FOLDER}/policy_net_{RESUME_STEP}.pt")
# target_net = DQN(n_observations, n_actions).to(device)
# target_net.load_state_dict(policy_net.state_dict())

# optimizer = optim.AdamW(
#     [*ast_net.parameters(), *policy_net.parameters()],
#     lr=LR,
#     amsgrad=True,
# )
# memory = ReplayMemory(REPLAY_MEMORY_SIZE)

# # Setup environment
# start = datetime.now()
# save_data_folder = Path("data") / RESUME_FOLDER

# logging.info("Setting up environment")
# engine = V8Engine()
# env = FuzzingEnv(
#     corpus,
#     subtrees,
#     engine,
#     total_coverage,
#     tokenizer,
#     INTERESTING_FOLDER / RESUME_FOLDER,
# )
# env.total_executions = total_executions
# env.total_actions = total_steps

# logging.info("Starting training")
# initial_coverage = env.total_coverage.coverage()

# try:
#     while total_steps < NUM_TRAINING_STEPS - RESUME_STEP:
#         state, info = env.reset()
#         done, truncated = False, False
#         episode_reward: list[float] = []
#         episode_action: list[tuple[int, str]] = []

#         while not done and not truncated:
#             action = epsilon_greedy(
#                 policy_net, state, ast_net, tokenizer, env, total_steps, device
#             )
#             episode_action.append((action, env._state.target_node.type))

#             next_state, reward, truncated, done, info = env.step(action)
#             episode_reward.append(reward)

#             total_steps += 1

#             memory.push(state, action, next_state, reward)
#             loss = optimise_model(
#                 policy_net,
#                 target_net,
#                 ast_net,
#                 tokenizer,
#                 optimizer,
#                 memory,
#                 device,
#                 total_steps,
#             )
#             soft_update_params(policy_net, target_net)

#             losses.append(loss)

#             state = next_state
#             if total_steps % 100 == 0:
#                 execution_coverage[
#                     (env.total_executions, total_steps)
#                 ] = env.total_coverage.coverage()

#             if total_steps % 1000 == 0:
#                 torch.save(ast_net, save_data_folder / f"ast_net_{total_steps}.pt")
#                 torch.save(
#                     policy_net, save_data_folder / f"policy_net_{total_steps}.pt"
#                 )
#                 torch.save(
#                     target_net, save_data_folder / f"target_net_{total_steps}.pt"
#                 )
#                 current_coverage = env.total_coverage.coverage()
#                 total_executions = env.total_executions

#                 with open(save_data_folder / f"run_data_{total_steps}.pkl", "wb") as f:
#                     pickle.dump(
#                         {
#                             "episode_actions": episode_actions,
#                             "episode_rewards": episode_rewards,
#                             "episode_coverage": episode_coverage,
#                             "execution_coverage": execution_coverage,
#                             "current_coverage": current_coverage,
#                             "total_steps": total_steps,
#                             "total_executions": total_executions,
#                             "running_time": datetime.now() - start,
#                             "losses": losses,
#                         },
#                         f,
#                     )

#         episode_coverage.append(env.total_coverage.coverage())
#         episode_rewards.append(episode_reward)
#         episode_actions.append(episode_action)


# except Exception as e:
#     traceback.print_exception(type(e), e, e.__traceback__)

# finally:
#     end = datetime.now()
#     episode_rewards_summed = [sum(episode) for episode in episode_rewards]

#     logging.info(f"Initial coverage: {initial_coverage:.5%}")
#     logging.info(
#         f"Finished with final coverage: {env.total_coverage} in {end - start}",
#     )
#     logging.info(
#         f"Coverage increase: {(env.total_coverage.coverage() - initial_coverage):.5%}"
#     )
#     logging.info(f"Average reward: {np.mean(episode_rewards_summed):.2f}")
#     logging.info(f"Total steps: {env.total_actions}")
#     logging.info(f"Total engine executions: {env.total_executions}")
