# JEFRL: JavaScript Engine Fuzzing using Reinforcement Learning

JEFRL fuzzes JavaScript engines by letting a Double DQN agent navigate and mutate the
AST of seed programs, rewarded by the engine coverage its mutations reach. See the
[report](write-up.pdf) for the design.

## Prerequisites

### Building V8

The engine is built in Docker (x64 Linux, which matches the toolchain V8 8.5 expects;
on Apple Silicon it runs under Rosetta):

```bash
cd engines/v8
docker build --platform linux/amd64 -t v8-build .
docker run --platform linux/amd64 -it -v "$PWD":/v8 v8-build            # out/fuzzbuild/d8
docker run --platform linux/amd64 -it -v "$PWD":/v8 -e BUILD=asan v8-build  # optional, for triaging crashes
sudo chown -R $USER:$USER . # to change ownership of the files created by docker back to the user
```

`v8_fuzzilli=true` adds Fuzzilli's coverage instrumentation and its REPRL mode to d8.
In REPRL mode one d8 process runs many inputs instead of being started for each one,
which is much faster. The fuzzing build leaves ASan out for speed; re-run crashes on
the `asan` build to get a report.

### Python

Python 3.11 with the dependencies in `pyproject.toml` (`poetry install`), or with uv:

```bash
uv venv -p 3.11 && uv pip install "esprima==4.0.1" "numpy<2" tqdm "transformers>=4.40,<5" \
    torch strenum "gymnasium>=0.28,<1" esutils escodegen pytest
```

Pre-training uses a CUDA GPU or Apple Silicon (MPS) if available. Fuzzing uses a frozen
model, so it runs fine on CPU.

### Where to run the fuzzer

On a Linux x64 machine, run everything directly with `--engine-mode reprl` (the default
picks it automatically).

On a Mac, d8 is a Linux binary that has to run in a container under Rosetta, where
PyTorch has no vector instructions and is 10-40x slower. So run only the engines in the
container, behind an engine server, and the fuzzer natively (using the GPU):

```bash
docker build --platform linux/amd64 -f Dockerfile.fuzzer -t jefrl-fuzzer .
docker run -d --name jefrl-engine --platform linux/amd64 --shm-size=2g \
    -p 127.0.0.1:9000:9000 -e PYTHONPATH=src -v "$PWD":/jefrl jefrl-fuzzer \
    python3 -m utils.engine_server --port 9000
```

Then pass `--engine-mode remote --engine-address localhost:9000` to the scripts below
instead of `--engine-executable`. The server starts one REPRL d8 per worker. The
container can also run the scripts directly (`docker run ... jefrl-fuzzer python3 src/...`),
which is fine for pre-processing but slow for anything using the model.

### Check the engine build

Before a long run, check that REPRL, coverage and crash detection work:

```bash
python3 src/check_engine.py --engine-executable engines/v8/v8/out/fuzzbuild/d8
```

This runs `fuzzilli('FUZZILLI_CRASH', n)` to make sure real crashes are detected, and
reports execs/s per worker.

## Running the fuzzer

JEFRL has a pre-processing step and then three stages: pre-training the AST Transformer,
(optionally) fine-tuning it, and fuzzing.

### Stage 0: Pre-processing

We use the [DIE-corpus](https://github.com/sslab-gatech/DIE-corpus), which includes
seeds from engine test suites and PoCs of past vulnerabilities:

```bash
git clone git@github.com:sslab-gatech/DIE-corpus.git corpus/DIE
python3 src/preprocessing.py --engine-executable engines/v8/v8/out/fuzzbuild/d8
```

Seeds are parsed (keeping V8 natives syntax such as `%OptimizeFunctionOnNextCall`),
sanitised and normalised, then executed with their test harness to record coverage.
Output goes to `data/`. Seeds that crash the engine are listed in
`data/seed_crashes.txt`: on an old engine these are known bugs, and a good sign that
crash detection works.

### Stage 1: Pre-training the AST Transformer

```bash
python3 src/pretraining.py --epochs 20
```

This creates a folder in `out/pretraining/` containing `model_best.pt` (lowest
validation loss) and a checkpoint per epoch.

### Stage 2: Fine-tuning the AST Transformer (optional)

Trains the DQN and the Transformer together, then keeps the Transformer weights:

```bash
python3 src/finetuning.py --pretraining-path out/pretraining/<run> --pretraining-step best
```

In the report fine-tuning made little difference to coverage, so the fuzzer can use the
pre-trained model directly.

### Stage 3: Fuzzing

```bash
python3 src/main.py --ast-model out/pretraining/<run>/model_best.pt \
    --engine-executable engines/v8/v8/out/fuzzbuild/d8 --workers 6 --max-hours 72
```

- `--workers N` runs N environments, each with its own engine process, sharing one
  corpus and coverage map. Use about one per core.
- `--max-hours` stops after a wall-clock time, for comparing against other fuzzers.
- `--engine-arg` passes extra flags to d8, e.g. `--engine-arg=--interrupt-budget=1024`
  to make V8 optimise functions sooner.
- `--engine-mode subprocess` starts a process per execution (slow; for engines without REPRL).
- `--seed-selection fast` picks the corpus program each episode starts from in proportion
  to how quickly its episodes execute (as AFL favours fast inputs), which stops the
  corpus drifting towards slow programs. The default, `uniform`, picks uniformly.

Runs are resumable. The networks, optimiser, replay buffer, corpus, coverage and
statistics are checkpointed every 30 minutes (`--checkpoint-minutes`) and whenever the
run stops, including on Ctrl-C or `kill`. To continue a stopped run:

```bash
python3 src/main.py --resume out/rl-training/<run> --engine-mode remote --engine-address localhost:9000 --workers 6
```

It keeps the original model, data, step limit and `--max-hours` (which counts the total
time across resumes), while engine options and the number of workers can change.

Results are written to `out/rl-training/<run>/`:

- `interesting/`: programs that increased coverage (`*_coverage.js`) or crashed the
  engine (`*_crash.js`, with its output in the matching `.txt`)
- `training/stats.jsonl`: one line per minute with coverage, execs/s, crashes, action
  counts and the distribution of execution errors
- `training/models/`: DQN checkpoints

### Random baseline

```bash
python3 src/baseline.py --engine-executable engines/v8/v8/out/fuzzbuild/d8 --workers 6 --max-hours 72
```

Picks uniformly among the valid actions, which is the fair comparison for the DQN agent.
`--unmasked` reproduces the original baseline, which also picks actions that cannot apply.

It records the same `stats.jsonl` and coverage curves as the DQN fuzzer, so
`python3 src/status.py out/baseline/<run>` works on it too, and it is checkpointed and
resumed the same way (`--resume out/baseline/<run>`). For a fair comparison, give both
the same `--workers`, `--timeout` and `--max-hours`, and run them one after the other
rather than at the same time.

### Tests

```bash
python3 -m pytest
```

The engine tests use `tests/utils/fake_d8.py`, a stand-in that speaks the REPRL protocol,
so they don't need a V8 build.

## Project Timeline

-   [x] Implement a way to run a JavaScript program through an engine from Python and receive outputs such as status, coverage, etc (Target: Late January)
-   [x] Design and implement a way to take a JavaScript Abstract Syntax Tree (AST) along with a mutation as input and return a mutated AST (Target: Mid-Late February)
-   [x] Implement Environment which can take an action from the RL agent, modify the current JavaScript AST based on this action, run the program through the engine, and return the reward along with the new state. (Target: Mid-Late March)
-   [x] Implement a DQN agent which takes a JavaScript AST as input and then outputs the desired action to be taken (Target: Early April)
-   [x] Make mutations smarter by taking into account the variable/functions in scope (Target: Mid April)
-   [x] Add more unit tests for each module due to errors arising during execution (Target: Late April)
-   [x] Come up with a comprehensive evaluation metric for the fuzzers to allow for easy comparision, eg. coverage in a certain time period of execution or number of crashes (Target: Early May)
-   [x] Implement a transformer model using similar technique to montage to tokenize/embed the AST directly instead of converting to code (Target: Late May)
-   [x] Execute fuzzer on several versions of JavaScript engines and use the evaluation metric to tune reward function and hyperparameters (Target: Early June)
-   [x] Compare results of RL agent with a random agent and other fuzzers in terms of bugs found in older versions and coverage gained per time period (Target: Early June)
-   [x] Write a report (Target: Mid June)
