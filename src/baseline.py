"""Random-agent baseline. By default it picks uniformly among the valid actions, which
is the fair comparison for the DQN agent (which also only picks valid actions).
--unmasked reproduces the original baseline, which picks from all actions.

Like the DQN fuzzer (main.py), runs are checkpointed and can be resumed with
--resume, and record the same statistics and coverage curves so the two can be
compared directly."""

import argparse
from collections import Counter
from datetime import datetime
import json
import logging
import os
from pathlib import Path
import pickle
import random
import signal
import sys
import threading
import time
from typing import Any

import numpy as np
from js_ast.analysis import strip_scopes
from rl.checkpoint import find_checkpoint
from rl.checkpoint import load_checkpoint
from rl.checkpoint import save_checkpoint
from rl.env import FuzzingEnv
from rl.env import MAX_FRAGMENT_SEQ_LEN
from rl.env import SharedFuzzingState
from rl.train import MAX_EPISODE_STEPS
from rl.train import MAX_MUTATIONS
from rl.train import NUM_TRAINING_STEPS
from transformer.tokenizer import ASTTokenizer

from utils.js_engine import Engine
from utils.js_engine import JSError
from utils.js_engine import TIMEOUT
from utils.logging import setup_logging
from utils.seed import setup_seeds


parser = argparse.ArgumentParser()

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
    "--engine-mode", choices=["auto", "reprl", "subprocess", "remote"], default="auto"
)
parser.add_argument("--engine-arg", action="append", default=[])
parser.add_argument("--timeout", type=float, default=TIMEOUT)
parser.add_argument("--workers", type=int, default=1)
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
    "--unmasked",
    action="store_true",
    help="Sample from all actions, including ones that cannot apply (the original baseline)",
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
    help="Resume the run in this folder (e.g. out/baseline/<run>) from its last checkpoint",
)
parser.add_argument(
    "--checkpoint-minutes",
    type=float,
    default=30,
    help="How often to checkpoint the run (it is also checkpointed when it stops)",
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
    default=Path("out/baseline/"),
    help="The path to the output directory where processed files will be stored",
)
parser.add_argument(
    "--engine-address",
    default="localhost:9000",
    help="host:port of an engine server (src/utils/engine_server.py) for --engine-mode remote",
)
args = parser.parse_args()

checkpoint_path = None
previous_hyperparameters: dict[str, Any] = {}
if args.resume is not None:
    checkpoint_path = find_checkpoint(args.resume / "training")
    if checkpoint_path is None:
        parser.error(f"No complete checkpoint found in {args.resume}")
    with open(args.resume / "hyperparameters.json") as f:
        previous_hyperparameters = json.load(f)
    # Continue with the same data and settings as the original run
    args.data_dir = Path(previous_hyperparameters["data_dir"])
    args.unmasked = not previous_hyperparameters["masked"]
    if args.num_steps is None:
        args.num_steps = previous_hyperparameters["num_training_steps"]
    if args.max_hours is None:
        args.max_hours = previous_hyperparameters["max_hours"]

if args.num_steps is None:
    # A time-limited run has no step limit
    args.num_steps = float("inf") if args.max_hours else NUM_TRAINING_STEPS

if args.seed_selection is None:
    args.seed_selection = previous_hyperparameters.get("seed_selection", "uniform")

if not args.data_dir.exists():
    raise ValueError(f"Data directory {args.data_dir} does not exist")

if args.engine_mode != "remote" and not args.engine_executable.exists():
    raise ValueError(f"Engine executable {args.engine_executable} does not exist")

os.makedirs(args.output, exist_ok=True)

# Output folders (a resumed run keeps writing to its original folder)
fuzz_start = datetime.now()
data_folder = args.resume or args.output / fuzz_start.strftime("%Y-%m-%dT%H-%M-%S")
interesting_tests_folder = data_folder / "interesting"
training_folder = data_folder / "training"
run_data_folder = training_folder / "run_data"
for folder in (interesting_tests_folder, run_data_folder):
    os.makedirs(folder, exist_ok=True)

# System setup
sys.setrecursionlimit(10000)
setup_logging(data_folder / "baseline.log")
seed = previous_hyperparameters.get("seed") or setup_seeds()

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

tokenizer = ASTTokenizer(vocab_data["vocab"], vocab_data["token_to_id"], MAX_FRAGMENT_SEQ_LEN)

num_workers = max(1, args.workers)
shared = SharedFuzzingState(data["corpus"], data["total_coverage"], seed_selection=args.seed_selection)
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
envs = [
    FuzzingEnv(
        shared,
        data["subtrees"],
        engine,
        tokenizer,
        interesting_tests_folder,
        max_mutations=MAX_MUTATIONS,
        max_steps=MAX_EPISODE_STEPS,
        worker_id=i,
    )
    for i, engine in enumerate(engines)
]
# numpy generators are not thread-safe, so each worker gets its own
rngs = [np.random.default_rng(seed + i) for i in range(num_workers)]

hyperparameters: dict[str, Any] = {
    "num_training_steps": args.num_steps,
    "max_hours": args.max_hours,
    "workers": num_workers,
    "masked": not args.unmasked,
    "engine": type(engines[0]).__name__,
    "engine_args": engines[0].args,
    "timeout": args.timeout,
    "seed_selection": args.seed_selection,
    "max_mutations": MAX_MUTATIONS,
    "max_episode_steps": MAX_EPISODE_STEPS,
    "seed": seed,
    "data_dir": str(args.data_dir),
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

initial_coverage = shared.total_coverage.coverage()
start_time = time.monotonic()
deadline = start_time + args.max_hours * 3600 if args.max_hours else None
elapsed_before = 0.0  # time spent in previous sessions of this run

counters_lock = threading.Lock()
total_steps = 0
step_errors = 0
stop = threading.Event()

# Coverage curves, sampled every 100 steps
execution_coverage: dict[int, float] = {}
action_coverage: dict[int, float] = {}
time_coverage: dict[float, float] = {}


def elapsed() -> float:
    """Wall-clock time of the run, including previous sessions if resumed."""
    return elapsed_before + time.monotonic() - start_time


def total_executions() -> int:
    return sum(env.total_executions for env in envs)


def worker(i: int):
    """Runs episodes on one environment until told to stop."""
    global total_steps, step_errors
    env = envs[i]
    rng = rngs[i]
    _, info = env.reset(seed=seed + i + total_steps)

    while not stop.is_set():
        if args.unmasked:
            action = int(rng.integers(env.action_space.n))
        else:
            action = int(rng.choice(np.flatnonzero(info["action_mask"])))

        try:
            _, _, terminated, truncated, info = env.step(action)
        except Exception:
            # A bug hit while mutating one program shouldn't end a multi-day run
            with counters_lock:
                step_errors += 1
            logging.exception(f"Error in environment {i}, starting a new episode")
            terminated, truncated = False, True

        with counters_lock:
            total_steps += 1

        if terminated or truncated:
            _, info = env.reset()


def write_stats():
    run_time = elapsed()
    executions = total_executions()
    exec_errors: Counter[str] = Counter()
    action_counts: Counter[str] = Counter()
    for env in envs:
        exec_errors.update(env.exec_errors)
        action_counts.update(env.action_counts)

    exec_time = sum(env.exec_time for env in envs)
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
        "failed_actions": sum(env.num_failed_actions for env in envs),
        "rollbacks": sum(env.num_rollbacks for env in envs),
        "coverage_rate": shared.coverage_rate,
        "step_errors": step_errors,
        "exec_errors": dict(exec_errors),
        "actions": dict(action_counts),
        "time_exec_s": round(exec_time, 1),
        "time_mutate_s": round(sum(env.action_time for env in envs), 1),
        "time_codegen_s": round(sum(env.code_gen_time for env in envs), 1),
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
                "current_coverage": shared.total_coverage.coverage(),
                "total_steps": total_steps,
                "total_executions": total_executions(),
                "running_time": datetime.now() - fuzz_start,
            },
            f,
        )


def save_run_checkpoint():
    start = time.perf_counter()
    corpus_state = shared.snapshot()
    state = {
        "total_steps": total_steps,
        "step_errors": step_errors,
        "elapsed": elapsed(),
        "execution_coverage": dict(execution_coverage),
        "action_coverage": dict(action_coverage),
        "time_coverage": dict(time_coverage),
        "env_stats": [env.stats_state() for env in envs],
        "rng": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "workers": [rng.bit_generator.state for rng in rngs],
        },
    }
    save_checkpoint(
        training_folder,
        state,
        corpus_state,
        {"steps": state["total_steps"], "coverage": corpus_state["total_coverage"].coverage()},
    )
    logging.info(
        f"Saved checkpoint at step {state['total_steps']} in {time.perf_counter() - start:.1f}s"
    )


if checkpoint_path is not None:
    logging.info(f"Resuming from {checkpoint_path}")
    state, corpus_state, _ = load_checkpoint(checkpoint_path)
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
    step_errors = state["step_errors"]
    elapsed_before = state["elapsed"]
    execution_coverage = state["execution_coverage"]
    action_coverage = state["action_coverage"]
    time_coverage = state["time_coverage"]
    initial_coverage = previous_hyperparameters["initial_coverage"]

    # With a different number of workers, pool the previous workers' statistics
    for i, stats in enumerate(state["env_stats"]):
        envs[i % num_workers].add_stats(stats)

    random.setstate(state["rng"]["python"])
    np.random.set_state(state["rng"]["numpy"])
    for rng, rng_state in zip(rngs, state["rng"]["workers"]):
        rng.bit_generator.state = rng_state

    if deadline is not None:
        deadline -= elapsed_before
    logging.info(
        f"Resumed at step {total_steps} after {elapsed_before / 3600:.2f}h, "
        f"corpus {len(shared)}, coverage {shared.total_coverage}"
    )

# Treat `kill` like Ctrl-C, so the run stops cleanly and saves a checkpoint
signal.signal(signal.SIGTERM, signal.default_int_handler)

logging.info(f"Initial coverage: {initial_coverage:.5%}, current: {shared.total_coverage}")
workers = [
    threading.Thread(target=worker, args=(i,), name=f"worker-{i}", daemon=True)
    for i in range(num_workers)
]
for thread in workers:
    thread.start()

last_stats_time = time.monotonic()
last_checkpoint_time = time.monotonic()
sampled_steps = total_steps
saved_steps = total_steps

try:
    while total_steps < args.num_steps and (
        deadline is None or time.monotonic() < deadline
    ):
        time.sleep(1)

        if total_steps // 100 > sampled_steps // 100:
            sampled_steps = total_steps
            coverage = shared.total_coverage.coverage()
            execution_coverage[total_executions()] = coverage
            action_coverage[total_steps] = coverage
            time_coverage[round(elapsed(), 1)] = coverage

        if time.monotonic() - last_stats_time > 60:
            last_stats_time = time.monotonic()
            write_stats()

        if total_steps // 10000 > saved_steps // 10000:
            saved_steps = total_steps
            save_run_data()

        if time.monotonic() - last_checkpoint_time > args.checkpoint_minutes * 60:
            save_run_checkpoint()
            last_checkpoint_time = time.monotonic()

        if not any(thread.is_alive() for thread in workers):
            raise RuntimeError("All worker threads have stopped")
except KeyboardInterrupt:
    logging.info("Stopping (interrupted), resume with --resume " + str(data_folder))
finally:
    stop.set()
    for thread in workers:
        thread.join(timeout=30)
    write_stats()
    save_run_data()
    save_run_checkpoint()
    for engine in engines:
        engine.close()
    logging.info(
        f"Finished: initial coverage {initial_coverage:.5%}, final coverage {shared.total_coverage}"
    )
