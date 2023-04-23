from itertools import count

import torch
from torch import optim
from transformers import RobertaConfig, RobertaModel, RobertaTokenizer
from js_ast.scope import Scope

from rl.dqn import DQN, ReplayMemory
from rl.env import FuzzingEnv, ProgramState
from rl.train import epsilon_greedy, optimise_model, soft_update_params
from utils.analysis import live_variable_analysis
from utils.js_engine import ExecutionData, V8Engine
from utils.loader import get_subtrees, load_corpus

# Environment setup
engine = V8Engine()
corpus = load_corpus(engine.get_corpus())
subtrees = get_subtrees(corpus)

# TODO: Check deepcopy issues
for ast in corpus:
    live_variable_analysis(ast, Scope())

corpus = [ProgramState(ast) for ast in corpus]
env = FuzzingEnv(corpus, subtrees, engine)


LR = 1e-4  # Learning rate of the AdamW optimizer
NUM_EPISODES = 1000  # Number of episodes to train the agent for


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

# import esprima

# from js_ast.nodes import CallExpression, Identifier, Literal, Node
# from js_ast.scope import Scope
# from utils.analysis import live_variable_analysis
# from utils.js_engine import V8Engine
# from utils.mutation import replace

# # code = """
# # function f(x) {
# #     return x + 1;
# # }

# # f(10);
# # """
# code = {
#     "type": "Program",
#     "sourceType": "script",
#     "body": [
#         {
#             "type": "FunctionDeclaration",
#             "id": {"type": "Identifier", "name": "f"},
#             "params": [{"type": "Identifier", "name": "x"}],
#             "generator": False,
#             "async": False,
#             "expression": False,
#             "body": {
#                 "type": "BlockStatement",
#                 "body": [
#                     {
#                         "type": "ReturnStatement",
#                         "argument": {
#                             "type": "BinaryExpression",
#                             "operator": "+",
#                             "left": {"type": "Identifier", "name": "x"},
#                             "right": {"type": "Literal", "value": 1, "raw": "1"},
#                         },
#                     }
#                 ],
#             },
#         },
#         {
#             "type": "FunctionDeclaration",
#             "id": {"type": "Identifier", "name": "g"},
#             "params": [{"type": "Identifier", "name": "x"}],
#             "generator": False,
#             "async": False,
#             "expression": False,
#             "body": {
#                 "type": "BlockStatement",
#                 "body": [
#                     {
#                         "type": "ReturnStatement",
#                         "argument": {
#                             "type": "BinaryExpression",
#                             "operator": "+",
#                             "left": {"type": "Identifier", "name": "x"},
#                             "right": {"type": "Literal", "value": 1, "raw": "1"},
#                         },
#                     }
#                 ],
#             },
#         },
#         {
#             "type": "ExpressionStatement",
#             "expression": {
#                 "type": "CallExpression",
#                 "callee": {"type": "Identifier", "name": "f"},
#                 "arguments": [{"type": "Literal", "value": 10, "raw": "10"}],
#             },
#         },
#     ],
# }

# tree = Node.from_dict(code)
# target = tree.body[2].expression
# live_variable_analysis(tree, Scope())

# subtrees = {
#     "CallExpression": [CallExpression(False, Identifier("g"), [Literal(1, "1")])]
# }
# # engine = V8Engine()
# # print(engine.execute(tree))
# print(tree)

# replace(subtrees, target)

# print(tree)
