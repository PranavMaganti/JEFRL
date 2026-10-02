import argparse
import contextlib
import copy
from collections import Counter
from datetime import datetime
import json
import logging
import os
from pathlib import Path
import pickle
import queue
import random
import shutil
import signal
import sys
import threading
import time

import numpy as np
from js_ast.analysis import strip_scopes
from rl.checkpoint import find_checkpoint
from rl.checkpoint import load_checkpoint
from rl.checkpoint import save_checkpoint
from rl.dqn import DQN
from rl.dqn import ReplayBuffer
from rl.env import FuzzingEnv
from rl.env import MAX_FRAGMENT_SEQ_LEN
from rl.env import SharedFuzzingState
from rl.fuzzing_action import FuzzingAction
from rl.train import BATCH_SIZE
from rl.train import EPS_DECAY_STEPS
from rl.train import EPS_END
from rl.train import EPS_START
from rl.train import epsilon
from rl.train import GAMMA
from rl.train import GRADIENT_CLIP
from rl.train import LR
from rl.train import MAX_EPISODE_STEPS
from rl.train import MAX_MUTATIONS
from rl.train import NUM_TRAINING_STEPS
from rl.train import optimise_model
from rl.train import REPLAY_MEMORY_SIZE
from rl.train import N_STEP
from rl.train import NStepAccumulator
from rl.train import PER_ALPHA
from rl.train import PER_BETA_START
from rl.train import PER_BETA_STEPS
from rl.train import STEPS_PER_UPDATE
from rl.train import select_actions
from rl.train import soft_update_params
from rl.train import StateEmbedder
from rl.train import TARGET_UPDATE
from rl.train import TAU
from rl.train import WARM_UP_STEPS
import torch
from torch import optim
from transformer.ast_transformer import get_ast_transformer_model
from transformers import get_constant_schedule_with_warmup

from utils.device import get_device
from utils.js_engine import Engine
from utils.js_engine import JSError
from utils.js_engine import TIMEOUT
from utils.logging import setup_logging
from utils.seed import setup_seeds


parser = argparse.ArgumentParser()

parser.add_argument(
    "--ast-model",
    type=Path,
    help="Path to the AST Transformer weights (from pre-training or fine-tuning)",
)
parser.add_argument(
    "--finetuning-path", type=Path, help="The path to the finetuning model folder"
)
parser.add_argument(
    "--finetuning-step", type=int, help="The step of the finetuning model to load"
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
    "--engine-mode",
    choices=["auto", "reprl", "subprocess", "remote"],
    default="auto",
    help="reprl keeps one engine process alive per worker (much faster), subprocess starts one per execution",
)
parser.add_argument(
    "--engine-arg",
    action="append",
    default=[],
    help="Extra flag to pass to the engine, can be repeated (e.g. --engine-arg=--interrupt-budget=1024)",
)
parser.add_argument(
    "--timeout", type=float, default=TIMEOUT, help="Execution timeout in seconds"
)
parser.add_argument(
    "--workers",
    type=int,
    default=1,
    help="Number of environments (each with its own engine process) stepped in parallel",
)
parser.add_argument(
    "--steps-per-update",
    type=float,
    default=STEPS_PER_UPDATE,
    help="Environment steps (across all workers) per DQN update",
)
parser.add_argument(
    "--batch-size", type=int, default=BATCH_SIZE, help="Transitions per DQN update"
)
parser.add_argument(
    "--replay-size", type=int, default=REPLAY_MEMORY_SIZE, help="Replay buffer capacity"
)
parser.add_argument(
    "--eps-end", type=float, default=EPS_END, help="Final exploration rate"
)
parser.add_argument(
    "--n-step", type=int, default=N_STEP, help="Steps per return (1 for one-step Q-learning)"
)
parser.add_argument(
    "--per-alpha",
    type=float,
    default=PER_ALPHA,
    help="Prioritised replay exponent (0 samples uniformly)",
)
parser.add_argument(
    "--per-beta-start",
    type=float,
    default=PER_BETA_START,
    help="Initial importance-sampling correction for prioritised replay (annealed to 1)",
)
parser.add_argument(
    "--per-beta-steps",
    type=int,
    default=PER_BETA_STEPS,
    help="Steps over which the importance-sampling correction reaches 1",
)
parser.add_argument(
    "--dueling",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Use a dueling Q-network",
)
parser.add_argument(
    "--actor-sync-updates",
    type=int,
    default=50,
    help="Copy the learner's policy to the actors every this many updates",
)
parser.add_argument(
    "--learner-threads",
    type=int,
    default=2,
    help="CPU threads for the learner's PyTorch operations (when learning on the CPU)",
)
parser.add_argument(
    "--reset-agent",
    action="store_true",
    help="When resuming, start with new networks and an empty replay buffer (keeping "
    "the corpus, coverage and statistics), e.g. after changing the agent's settings",
)
parser.add_argument(
    "--eps-decay-steps",
    type=int,
    default=EPS_DECAY_STEPS,
    help="Steps over which exploration decays linearly from 1 to --eps-end",
)
parser.add_argument(
    "--num-steps",
    type=int,
    default=None,
    help=f"Stop after this many actions (default {NUM_TRAINING_STEPS})",
)
parser.add_argument(
    "--max-hours",
    type=float,
    default=None,
    help="Stop after this much wall-clock time (in total, across resumes)",
)
parser.add_argument(
    "--seed-selection",
    choices=["uniform", "fast"],
    default=None,
    help="How to pick the corpus program each episode starts from: uniformly, or "
    "favouring programs whose episodes execute quickly (like AFL). Default: uniform, or "
    "the resumed run's setting",
)
parser.add_argument(
    "--resume",
    type=Path,
    default=None,
    help="Resume the run in this folder (e.g. out/rl-training/<run>) from its last checkpoint",
)
parser.add_argument(
    "--checkpoint-minutes",
    type=float,
    default=30,
    help="How often to checkpoint the run (it is also checkpointed when it stops)",
)
parser.add_argument(
    "--device",
    choices=["auto", "cpu", "mps", "cuda"],
    default="auto",
    help="Device for the AST Transformer and DQN (auto: CUDA, then MPS, then CPU)",
)
parser.add_argument(
    "--data-dir",
    type=Path,
    default=Path("data/"),
    help="The path to which the vocabulary and pretraining data was saved",
)
parser.add_argument(
    "--output",
    type=Path,
    default=Path("out/rl-training/"),
    help="The path to the output directory where processed files will be stored",
)

parser.add_argument(
    "--engine-address",
    default="localhost:9000",
    help="host:port of an engine server (src/utils/engine_server.py) for --engine-mode remote",
)
args = parser.parse_args()

checkpoint_path = None
previous_hyperparameters: dict = {}
if args.resume is not None:
    checkpoint_path = find_checkpoint(args.resume / "training")
    if checkpoint_path is None:
        parser.error(f"No complete checkpoint found in {args.resume}")
    with open(args.resume / "hyperparameters.json") as f:
        previous_hyperparameters = json.load(f)
    # Continue with the same model and data as the original run
    args.ast_model = args.resume / "training" / "models" / "ast_net.pt"
    args.data_dir = Path(previous_hyperparameters["data_dir"])
    if args.num_steps is None:
        args.num_steps = previous_hyperparameters["num_training_steps"]
    if args.max_hours is None:
        args.max_hours = previous_hyperparameters["max_hours"]

if args.num_steps is None:
    # A time-limited run has no step limit
    args.num_steps = float("inf") if args.max_hours else NUM_TRAINING_STEPS

if args.seed_selection is None:
    args.seed_selection = previous_hyperparameters.get("seed_selection", "uniform")

if args.ast_model is None:
    if args.finetuning_path is None or args.finetuning_step is None:
        parser.error("pass --ast-model, or --finetuning-path and --finetuning-step")
    args.ast_model = (
        args.finetuning_path / f"training/models/ast_net_{args.finetuning_step}.pt"
    )

if not args.data_dir.exists():
    raise ValueError(f"Data directory {args.data_dir} does not exist")

if not args.ast_model.exists():
    raise ValueError(f"AST Transformer weights {args.ast_model} do not exist")

if args.engine_mode != "remote" and not args.engine_executable.exists():
    raise ValueError(f"Engine executable {args.engine_executable} does not exist")

os.makedirs(args.output, exist_ok=True)

num_workers = max(1, args.workers)

# Output folders (a resumed run keeps writing to its original folder)
fuzz_start = datetime.now()
data_folder = args.resume or args.output / fuzz_start.strftime("%Y-%m-%dT%H-%M-%S")
interesting_tests_folder = data_folder / "interesting"
training_folder = data_folder / "training"
models_folder = training_folder / "models"
run_data_folder = training_folder / "run_data"
for folder in (interesting_tests_folder, models_folder, run_data_folder):
    os.makedirs(folder, exist_ok=True)

# System setup
sys.setrecursionlimit(10000)
setup_logging(data_folder / "fuzzer.log")
seed = setup_seeds()

# Load preprocessed data
logging.info("Loading preprocessed data")
with open(args.data_dir / "corpus.pkl", "rb") as f:
    data = pickle.load(f)

with open(args.data_dir / "vocab_data.pkl", "rb") as f:
    vocab_data = pickle.load(f)

# The corpus and subtree pool are only ever copied, and copies are re-analysed, so
# drop the scope information stored during preprocessing to save memory
stripped = strip_scopes(
    [state.root for state in data["corpus"]]
    + [node for nodes in data["subtrees"].values() for node in nodes]
)
logging.info(f"Dropped stored scopes from {stripped:,} AST nodes")

corpus = data["corpus"]
subtrees = data["subtrees"]
total_coverage = data["total_coverage"]

vocab = vocab_data["vocab"]
token_to_id = vocab_data["token_to_id"]

device = get_device() if args.device == "auto" else torch.device(args.device)
# The learner (DQN updates and replay buffer) runs on the CPU unless there's a CUDA
# GPU. An Apple Silicon GPU can only be used by one thread at a time, and sharing it
# with the learner left the actors waiting for it most of the time.
learner_device = device if device.type == "cuda" else torch.device("cpu")
if learner_device.type == "cpu":
    torch.set_num_threads(args.learner_threads)
logging.info(f"Embedding states on {device}, learning on {learner_device}")

tokenizer, ast_net, config = get_ast_transformer_model(
    vocab, token_to_id, args.ast_model, device
)
ast_net.eval()
for param in ast_net.parameters():
    param.requires_grad = False
embedder = StateEmbedder(ast_net, tokenizer, device)

if args.resume is None:
    shutil.copyfile(args.ast_model, models_folder / "ast_net.pt")

# Initialise policy and target networks
logging.info("Initialising policy and target networks")
n_actions = len(FuzzingAction)
# Input size to the DQN is the size of the AST Transformer hidden state * 2 (target and context)
n_observations = config.hidden_size * 2

policy_net = DQN(n_observations, n_actions, dueling=args.dueling).to(learner_device)
target_net = DQN(n_observations, n_actions, dueling=args.dueling).to(learner_device)
target_net.load_state_dict(policy_net.state_dict())
# The actors choose actions with their own copy of the policy, synced from the
# learner every --actor-sync-updates updates (as in Ape-X), so acting never waits
# for training
actor_net = DQN(n_observations, n_actions, dueling=args.dueling).to(learner_device)
actor_net.load_state_dict(policy_net.state_dict())
for net in (target_net, actor_net):
    for param in net.parameters():
        param.requires_grad = False

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
# Constant after warm-up, so runs can be open-ended (e.g. limited by --max-hours)
lr_scheduler = get_constant_schedule_with_warmup(optimizer, WARM_UP_STEPS)
memory = ReplayBuffer(
    args.replay_size, n_observations, n_actions, learner_device, alpha=args.per_alpha
)

logging.info(f"Setting up {num_workers} environment(s)")
shared = SharedFuzzingState(corpus, total_coverage, seed_selection=args.seed_selection)
engines = [
    Engine.get_engine(
        args.engine_name,
        args.engine_executable,
        args.engine_mode,
        args.timeout,
        args.engine_arg,
        address=args.engine_address,
    )
    for _ in range(num_workers)
]
logging.info(f"Engine: {type(engines[0]).__name__}")
envs = [
    FuzzingEnv(
        shared,
        subtrees,
        engine,
        tokenizer,
        interesting_tests_folder,
        max_mutations=MAX_MUTATIONS,
        max_steps=MAX_EPISODE_STEPS,
        worker_id=i,
    )
    for i, engine in enumerate(engines)
]

hyperparameters = {
    "num_training_steps": args.num_steps,
    "max_hours": args.max_hours,
    "workers": num_workers,
    "steps_per_update": args.steps_per_update,
    "engine": type(engines[0]).__name__,
    "engine_args": engines[0].args,
    "timeout": args.timeout,
    "replay_memory_size": args.replay_size,
    "learning_rate": LR,
    "warm_up_steps": WARM_UP_STEPS,
    "max_fragment_seq_len": MAX_FRAGMENT_SEQ_LEN,
    "seed_selection": args.seed_selection,
    "max_mutations": MAX_MUTATIONS,
    "max_episode_steps": MAX_EPISODE_STEPS,
    "eps_start": EPS_START,
    "eps_end": args.eps_end,
    "eps_decay_steps": args.eps_decay_steps,
    "eps_schedule": "linear",
    "n_step": args.n_step,
    "per_alpha": args.per_alpha,
    "per_beta_start": args.per_beta_start,
    "per_beta_steps": args.per_beta_steps,
    "dueling": args.dueling,
    "gamma": GAMMA,
    "batch_size": args.batch_size,
    "tau": TAU,
    "target_update": TARGET_UPDATE,
    "gradient_clip": GRADIENT_CLIP,
    "seed": seed,
    "ast_model": str(args.ast_model),
    "data_dir": str(args.data_dir),
    "device": str(device),
    "learner_device": str(learner_device),
    "actor_sync_updates": args.actor_sync_updates,
    "initial_corpus_size": len(shared),
    "initial_coverage": shared.total_coverage.coverage(),
}
if args.resume is not None:
    # Keep the original settings, and record what each resume ran with
    resumes = previous_hyperparameters.get("resumes", [])
    resumes.append({**hyperparameters, "resumed_at": fuzz_start.isoformat()})
    hyperparameters = {**previous_hyperparameters, "resumes": resumes}
with open(data_folder / "hyperparameters.json", "w") as f:
    json.dump(hyperparameters, f, indent=2)


# Each worker (actor) runs its own environment loop in a thread, so a slow execution
# only holds up its own worker. Actors share the policy for choosing actions and
# hand their transitions to the learner (the main thread), which trains the DQN.
# `gpu_lock` guards the embedder (and its cache): GPU work must never be issued from
# two threads at once, which crashes MPS. `actor_net_lock` guards the actors' copy
# of the policy while it is synced. The learner's networks and replay buffer are
# only used by the learner (main) thread.
gpu_lock = threading.Lock()
actor_net_lock = threading.Lock()
transitions: queue.Queue[tuple] = queue.Queue()
stop = threading.Event()

counters_lock = threading.Lock()
total_steps = 0
step_errors = 0
episode_returns: list[float] = []


def sync_actor_net():
    with actor_net_lock:
        actor_net.load_state_dict(policy_net.state_dict())


def current_epsilon() -> float:
    return epsilon(total_steps, args.eps_end, args.eps_decay_steps)


def current_beta() -> float:
    """Importance-sampling correction, annealed linearly to 1 (full correction)."""
    progress = min(1.0, learned_steps / max(1, args.per_beta_steps))
    return args.per_beta_start + (1.0 - args.per_beta_start) * progress


# Per-actor time breakdown (seconds): waiting for the GPU lock, holding it (GPU
# work), and stepping the environment (mutating, generating code and executing)
actor_times = [{"lock_wait": 0.0, "gpu": 0.0, "env_step": 0.0} for _ in range(num_workers)]


def embed_states(i: int, observations: list) -> torch.Tensor:
    with timed_gpu_lock(i):
        return embedder(observations).to(learner_device)


@contextlib.contextmanager
def timed_gpu_lock(i: int):
    requested = time.perf_counter()
    with gpu_lock:
        acquired = time.perf_counter()
        try:
            yield
        finally:
            actor_times[i]["lock_wait"] += acquired - requested
            actor_times[i]["gpu"] += time.perf_counter() - acquired


def actor(i: int):
    global total_steps, step_errors
    env = envs[i]
    obs, info = env.reset(seed=seed + i)
    mask = info["action_mask"]
    state_embedding = embed_states(i, [obs])
    episode_return = 0.0
    n_step = NStepAccumulator(args.n_step, GAMMA)

    while not stop.is_set():
        with actor_net_lock:
            action = int(
                select_actions(
                    actor_net,
                    state_embedding,
                    torch.from_numpy(mask).to(learner_device).unsqueeze(0),
                    current_epsilon(),
                ).item()
            )

        step_start = time.perf_counter()
        try:
            next_obs, reward, terminated, truncated, info = env.step(action)
        except Exception:
            # A bug hit while mutating one program shouldn't end a multi-day run
            with counters_lock:
                step_errors += 1
            logging.exception(f"Error in environment {i}, starting a new episode")
            next_obs, info = env.reset()
            reward, terminated, truncated = 0.0, False, True

        actor_times[i]["env_step"] += time.perf_counter() - step_start

        # Truncation is treated as the end of the episode too: the step count isn't
        # part of the state and the end-of-episode penalty assumes episodes end there
        done = bool(terminated or truncated)
        next_mask = info["action_mask"]
        next_embedding = embed_states(i, [next_obs])

        for transition in n_step.add(
            state_embedding, action, float(reward), next_embedding, done, next_mask
        ):
            transitions.put(transition)
        episode_return += reward

        with counters_lock:
            total_steps += 1
            if done:
                episode_returns.append(episode_return)

        if done:
            episode_return = 0.0
            next_obs, info = env.reset()
            next_mask = info["action_mask"]
            next_embedding = embed_states(i, [next_obs])

        obs, mask, state_embedding = next_obs, next_mask, next_embedding


logging.info("Starting training")
start_time = time.monotonic()
deadline = start_time + args.max_hours * 3600 if args.max_hours else None
initial_coverage = shared.total_coverage.coverage()

# Coverage curves, sampled every 100 steps
execution_coverage: dict[int, float] = {}
action_coverage: dict[int, float] = {}
time_coverage: dict[float, float] = {}

recent_losses: list[torch.Tensor] = []
optimisation_time = 0.0
last_stats_time = start_time


def elapsed() -> float:
    """Wall-clock time of the run, including previous sessions if resumed."""
    return elapsed_before + time.monotonic() - start_time


def total_executions() -> int:
    return sum(env.total_executions for env in envs)


def write_stats():
    run_time = elapsed()
    executions = total_executions()
    exec_errors: Counter[str] = Counter()
    action_counts: Counter[str] = Counter()
    for env in envs:
        exec_errors.update(env.exec_errors)
        action_counts.update(env.action_counts)

    exec_time = sum(env.exec_time for env in envs)
    mean_loss = (
        torch.stack(recent_losses[-1000:]).mean().item() if recent_losses else None
    )
    stats = {
        "elapsed_s": round(run_time, 1),
        "steps": total_steps,
        "executions": executions,
        "execs_per_s": round(executions / run_time, 2) if run_time > 0 else 0,
        # Average number of engines busy executing (at most --workers)
        "engine_utilisation": round(exec_time / run_time, 2) if run_time > 0 else 0,
        "coverage": shared.total_coverage.coverage(),
        "coverage_gain": shared.total_coverage.coverage() - initial_coverage,
        "corpus_size": len(shared),
        "corpus_added": shared.num_added,
        "corpus_removed": shared.num_removed,
        "crashes": shared.num_crashes,
        "epsilon": current_epsilon(),
        "mean_return_last_100": float(np.mean(episode_returns[-100:])) if episode_returns else None,
        "mean_loss_last_1000": mean_loss,
        "failed_actions": sum(env.num_failed_actions for env in envs),
        "rollbacks": sum(env.num_rollbacks for env in envs),
        "coverage_rate": shared.coverage_rate,
        "step_errors": step_errors,
        "exec_errors": dict(exec_errors),
        "actions": dict(action_counts),
        "time_exec_s": round(exec_time, 1),
        "time_mutate_s": round(sum(env.action_time for env in envs), 1),
        "time_codegen_s": round(sum(env.code_gen_time for env in envs), 1),
        "time_optimise_s": round(optimisation_time, 1),
        # Learner health: queued transitions not yet learned from, and the achieved
        # number of samples per transition (target: batch_size / steps_per_update)
        "learner_queue": transitions.qsize(),
        "replay_ratio": round(num_updates * args.batch_size / max(1, learned_steps), 2),
        "skipped_updates": int(skipped_updates),
        "time_timeouts_s": round(sum(env.timeout_time for env in envs), 1),
        "time_reset_s": round(sum(env.reset_time for env in envs), 1),
        # Summed over actors: compare with elapsed_s * workers
        "actor_time_s": {
            key: round(sum(times[key] for times in actor_times), 1)
            for key in actor_times[0]
        },
        "embedding_cache_hit_rate": round(
            embedder.hits / max(1, embedder.hits + embedder.misses), 3
        ),
    }
    with open(training_folder / "stats.jsonl", "a") as f:
        f.write(json.dumps(stats) + "\n")

    logging.info(
        f"steps={total_steps} execs={executions} ({stats['execs_per_s']}/s, "
        f"{stats['engine_utilisation']} engines busy) coverage={shared.total_coverage} "
        f"corpus={len(shared)} crashes={shared.num_crashes}"
    )


def save_run_data():
    with open(run_data_folder / f"{total_steps}.pkl", "wb") as f:
        pickle.dump(
            {
                "execution_coverage": execution_coverage,
                "action_coverage": action_coverage,
                "time_coverage": time_coverage,
                "episode_returns": episode_returns,
                "current_coverage": shared.total_coverage.coverage(),
                "total_steps": total_steps,
                "total_executions": total_executions(),
                "running_time": datetime.now() - fuzz_start,
            },
            f,
        )


learned_steps = 0  # transitions added to the replay buffer
elapsed_before = 0.0  # time spent in previous sessions of this run


def training_state(include_memory: bool) -> dict:
    # Copies, so the checkpoint can be written after releasing the model lock
    return {
        "policy_net": copy.deepcopy(policy_net.state_dict()),
        "target_net": copy.deepcopy(target_net.state_dict()),
        "optimizer": copy.deepcopy(optimizer.state_dict()),
        "lr_scheduler": copy.deepcopy(lr_scheduler.state_dict()),
        "memory": memory.state_dict() if include_memory else None,
        "total_steps": total_steps,
        "learned_steps": learned_steps,
        "step_errors": step_errors,
        "elapsed": elapsed(),
        "episode_returns": episode_returns[-100000:],
        "execution_coverage": execution_coverage,
        "action_coverage": action_coverage,
        "time_coverage": time_coverage,
        "env_stats": [env.stats_state() for env in envs],
        "rng": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
        },
    }


def save_run_checkpoint(include_memory: bool = False):
    """Checkpoints the run. The replay buffer (up to several GB) is only included
    when the run stops: copying it every half hour pushed the machine into swap. A
    crash loses the buffer (which refills) but not the agent or the corpus."""
    start = time.perf_counter()
    corpus_state = shared.snapshot()
    # Only the learner thread uses its networks and buffer, so this is consistent
    state = training_state(include_memory)
    save_checkpoint(
        training_folder,
        state,
        corpus_state,
        {"steps": state["total_steps"], "coverage": corpus_state["total_coverage"].coverage()},
    )
    logging.info(
        f"Saved checkpoint at step {total_steps} in {time.perf_counter() - start:.1f}s"
    )


if checkpoint_path is not None:
    logging.info(f"Resuming from {checkpoint_path}")
    state, corpus_state, info = load_checkpoint(checkpoint_path)
    if args.reset_agent:
        logging.info("Starting with a new agent (networks and replay buffer)")
    else:
        try:
            policy_net.load_state_dict(state["policy_net"])
            target_net.load_state_dict(state["target_net"])
            actor_net.load_state_dict(state["policy_net"])
        except RuntimeError as e:
            raise SystemExit(
                f"The checkpoint's networks don't match these settings ({e}). "
                "Use --reset-agent to continue the run with a new agent."
            )
        optimizer.load_state_dict(state["optimizer"])
        lr_scheduler.load_state_dict(state["lr_scheduler"])
        if state["memory"] is None:
            logging.info("Checkpoint has no replay buffer (not a clean stop), starting empty")
        elif "discounts" in state["memory"]:
            memory.load_state_dict(state["memory"])
        else:
            logging.info("Checkpoint's replay buffer predates n-step returns, starting empty")
    shared.restore(corpus_state)
    strip_scopes([entry.state.root for entry in shared.entries])
    # Programs added by executions which timed out or threw (before these were
    # excluded). The seeds all run cleanly, so only programs the fuzzer added go.
    pruned = shared.prune(
        lambda state: state.exec_data is not None
        and state.exec_data.error != JSError.NoError
    )
    if pruned:
        logging.info(f"Removed {pruned} programs which time out or throw from the corpus")

    total_steps = state["total_steps"]
    learned_steps = state["learned_steps"]
    step_errors = state["step_errors"]
    elapsed_before = state["elapsed"]
    episode_returns = state["episode_returns"]
    execution_coverage = state["execution_coverage"]
    action_coverage = state["action_coverage"]
    time_coverage = state["time_coverage"]
    initial_coverage = previous_hyperparameters["initial_coverage"]

    # With a different number of workers, pool the previous workers' statistics
    env_stats = state["env_stats"]
    for i, stats in enumerate(env_stats):
        envs[i % num_workers].add_stats(stats)

    random.setstate(state["rng"]["python"])
    np.random.set_state(state["rng"]["numpy"])
    torch.set_rng_state(state["rng"]["torch"])

    if deadline is not None:
        deadline -= elapsed_before
    logging.info(
        f"Resumed at step {total_steps} after {elapsed_before / 3600:.2f}h, "
        f"corpus {len(shared)}, coverage {shared.total_coverage}"
    )


# Treat `kill` like Ctrl-C, so the run stops cleanly and saves a checkpoint
signal.signal(signal.SIGTERM, signal.default_int_handler)

actors = [
    threading.Thread(target=actor, args=(i,), name=f"actor-{i}", daemon=True)
    for i in range(num_workers)
]
for thread in actors:
    thread.start()

updates_due = 0.0
num_updates = 0
skipped_updates = 0.0
MAX_PENDING_UPDATES = 8
last_checkpoint_time = time.monotonic()

try:
    while total_steps < args.num_steps and (
        deadline is None or time.monotonic() < deadline
    ):
        # Collect the transitions the actors produced since the last iteration
        # Take everything the actors have queued, so experience never goes stale
        batch = []
        try:
            batch.append(transitions.get(timeout=0.5))
            while len(batch) < 50000:
                batch.append(transitions.get_nowait())
        except queue.Empty:
            pass

        if batch:
            states, actions, rewards, next_states, dones, next_masks, discounts = zip(*batch)
            memory.push(
                torch.cat(states),
                torch.tensor(actions),
                torch.tensor(rewards),
                torch.cat(next_states),
                torch.tensor(dones),
                torch.from_numpy(np.stack(next_masks)),
                torch.tensor(discounts),
            )

        prev_steps = learned_steps
        learned_steps += len(batch)

        updates_due += len(batch) / args.steps_per_update
        # If the learner can't keep up with the actors, skip the excess updates
        # rather than falling behind: the replay ratio then adapts to the CPU time
        # available (reported as replay_ratio in the stats)
        if updates_due > MAX_PENDING_UPDATES:
            skipped_updates += updates_due - MAX_PENDING_UPDATES
            updates_due = MAX_PENDING_UPDATES
        start = time.perf_counter()
        while updates_due >= 1:
            updates_due -= 1
            loss = optimise_model(
                policy_net,
                target_net,
                optimizer,
                lr_scheduler,
                memory,
                batch_size=args.batch_size,
                beta=current_beta(),
            )
            if loss is not None:
                recent_losses.append(loss)
                num_updates += 1
                if num_updates % args.actor_sync_updates == 0:
                    sync_actor_net()
        optimisation_time += time.perf_counter() - start
        if len(recent_losses) > 2000:
            recent_losses = recent_losses[-1000:]

        if learned_steps // TARGET_UPDATE > prev_steps // TARGET_UPDATE:
            soft_update_params(policy_net, target_net)

        if learned_steps // 100 > prev_steps // 100:
            coverage = shared.total_coverage.coverage()
            execution_coverage[total_executions()] = coverage
            action_coverage[learned_steps] = coverage
            time_coverage[round(elapsed(), 1)] = coverage

        if time.monotonic() - last_stats_time > 60:
            last_stats_time = time.monotonic()
            write_stats()

        if learned_steps // 10000 > prev_steps // 10000:
            save_run_data()
            torch.save(policy_net.state_dict(), models_folder / f"policy_net_{learned_steps}.pt")
            torch.save(target_net.state_dict(), models_folder / f"target_net_{learned_steps}.pt")

        if time.monotonic() - last_checkpoint_time > args.checkpoint_minutes * 60:
            save_run_checkpoint()
            last_checkpoint_time = time.monotonic()

        if not any(thread.is_alive() for thread in actors):
            raise RuntimeError("All actor threads have stopped")
except KeyboardInterrupt:
    logging.info("Stopping (interrupted), resume with --resume " + str(data_folder))
finally:
    stop.set()
    for thread in actors:
        thread.join(timeout=30)
    write_stats()
    save_run_data()
    save_run_checkpoint(include_memory=True)
    torch.save(policy_net.state_dict(), models_folder / "policy_net_final.pt")
    for engine in engines:
        engine.close()
    logging.info(
        f"Finished: initial coverage {initial_coverage:.5%}, final coverage {shared.total_coverage}"
    )
