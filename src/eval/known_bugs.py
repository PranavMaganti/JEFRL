"""Measures how close fuzzing runs get to known bugs in the engine under test.

V8 lands a regression test with almost every bug fix. Tests added after the
engine's version was branched are proofs of concept for bugs present in it, so
the ones that crash the engine (with exactly the fuzzer's flags) form a ground
truth set. For each, the coverage edges specific to the bug's program are those it
reaches beyond the test harness and the engine's crash handling, and a run's
progress towards the bug is the share of those edges, among the ones the seed
corpus misses, that the run has reached.

    # Once per engine build (needs an engine, e.g. in the fuzzer container or via
    # --engine-mode remote):
    python3 src/eval/known_bugs.py prepare --since 2020-06-24 --until 2021-06-24

    # Any number of runs (DQN or baseline), no engine needed:
    python3 src/eval/known_bugs.py evaluate out/rl-training/<run> out/baseline/<run>

The tests are only ever used for evaluation: never add them to the corpus.
"""

import argparse
import json
from pathlib import Path
import pickle
import re
import subprocess
import sys

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.setrecursionlimit(10000)

from utils.js_engine import Engine  # noqa: E402


HARNESS = Path("corpus/libs/v8.js")
TEST_DIRS = ["test/mjsunit/regress/", "test/mjsunit/compiler/"]


def bits(coverage) -> np.ndarray:
    return np.unpackbits(coverage.edges).astype(bool)


def extract_tests(v8: Path, since: str, until: str, out: Path) -> list[dict]:
    """Regression tests added to V8's main branch between two dates."""
    base = subprocess.run(
        ["git", "-c", "safe.directory=*", "-C", str(v8), "merge-base", "HEAD", "origin/main"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    log = subprocess.run(
        ["git", "-c", "safe.directory=*", "-C", str(v8), "log", f"{base}..origin/main", f"--since={since}",
         f"--until={until}", "--diff-filter=A", "--name-only",
         "--format=COMMIT %H %ad", "--date=short", "--", *TEST_DIRS],
        capture_output=True, text=True, check=True,
    ).stdout

    tests: dict[str, dict] = {}
    commit = date = ""
    for line in log.splitlines():
        if line.startswith("COMMIT "):
            _, commit, date = line.split()
        elif line.endswith(".js"):
            name = line.replace("test/mjsunit/", "").replace("/", "__")
            source = subprocess.run(
                ["git", "-c", "safe.directory=*", "-C", str(v8), "show", f"{commit}:{line}"],
                capture_output=True, text=True,
            ).stdout
            (out / name).write_text(source)
            # Newest first in git log, so the first commit seen is the latest version
            tests.setdefault(name, {"name": name, "commit": commit, "date": date, "path": line})
    return list(tests.values())


def prepare(args: argparse.Namespace):
    out = args.out
    tests_dir = out / "tests"
    tests_dir.mkdir(parents=True, exist_ok=True)

    tests = extract_tests(args.v8, args.since, args.until, tests_dir)
    print(f"Extracted {len(tests)} regression tests added {args.since} to {args.until}")

    def engine() -> Engine:
        return Engine.get_engine(
            "v8", args.engine_executable, args.engine_mode, args.timeout,
            address=args.engine_address,
        )

    def run(code: str, lib=HARNESS):
        # A fresh engine per program, so no state carries over between them
        with engine() as e:
            return e.execute_text(code, lib)

    harness = bits(run("").coverage)
    # Edges of the engine's crash handling (fatal error and CHECK/DCHECK reports,
    # signal handlers, stack printing), which only crashing programs reach
    crash_edges = np.zeros_like(harness)
    for crash_type in range(3):
        data = run(f"fuzzilli('FUZZILLI_CRASH', {crash_type});")
        if not data.is_crash():
            sys.exit(f"FUZZILLI_CRASH {crash_type} did not crash: is the engine built with v8_fuzzilli?")
        crash_edges |= bits(data.coverage) & ~harness
    print(f"{crash_edges.sum()} crash-handling edges excluded")

    targets, edge_sets = [], {}
    for test in tests:
        data = run((tests_dir / test["name"]).read_text())
        if data is None or not data.is_crash():
            continue
        edges = bits(data.coverage) & ~harness & ~crash_edges
        report = [l for l in data.err.splitlines() if "Fatal error" in l or "Check failed" in l]
        targets.append({**test, "edges": int(edges.sum()), "crash": " | ".join(report[:2])})
        edge_sets[test["name"]] = np.packbits(edges)
        print(f"  crashes: {test['name']} ({test['date']}, {edges.sum()} edges)")

    np.savez_compressed(out / "edges.npz", **edge_sets)
    (out / "targets.json").write_text(json.dumps(targets, indent=1))
    print(f"\n{len(targets)} of {len(tests)} tests crash the engine with the fuzzer's flags")


def run_coverage(run: Path) -> tuple[np.ndarray, np.ndarray]:
    """A run's current total coverage (from its checkpoint) and its seeds' coverage."""
    from rl.checkpoint import find_checkpoint

    checkpoint = find_checkpoint(run / "training")
    if checkpoint is None:
        sys.exit(f"No checkpoint in {run}")
    with open(checkpoint / "corpus.pkl", "rb") as f:
        total = bits(pickle.load(f)["total_coverage"])

    hyperparameters = json.loads((run / "hyperparameters.json").read_text())
    with open(Path(hyperparameters["data_dir"]) / "corpus.pkl", "rb") as f:
        seeds = bits(pickle.load(f)["total_coverage"])
    return total, seeds


def evaluate(args: argparse.Namespace):
    targets = json.loads((args.out / "targets.json").read_text())
    stored = np.load(args.out / "edges.npz")

    results = {}
    seeds = None
    for run in args.runs:
        total, run_seeds = run_coverage(run)
        if seeds is not None and not np.array_equal(seeds, run_seeds):
            print(f"Warning: {run} started from different seeds")
        seeds = run_seeds
        results[run] = total

    header = f"{'known bug':42s} {'fixed':10s} {'edges':>6s} {'seeds miss':>10s}"
    for run in args.runs:
        header += f"  {run.name[:19]:>19s}"
    print(header)

    totals = {run: 0 for run in args.runs}
    total_missed = 0
    rows = []
    for target in targets:
        edges = np.unpackbits(stored[target["name"]])[: len(seeds)].astype(bool)
        missed = edges & ~seeds
        total_missed += missed.sum()
        reached = {run: int((missed & results[run]).sum()) for run in args.runs}
        for run in args.runs:
            totals[run] += reached[run]
        rows.append((target, int(edges.sum()), int(missed.sum()), reached))

    first = args.runs[0]
    for target, n, missed, reached in sorted(rows, key=lambda r: -r[3][first] / max(1, r[2])):
        line = f"{target['name']:42s} {target['date']:10s} {n:6d} {missed:10d}"
        for run in args.runs:
            line += f"  {reached[run]:6d} ({reached[run] / max(1, missed):4.0%})      "
        print(line)

    print()
    for run in args.runs:
        gained = int((results[run] & ~seeds).sum())
        print(
            f"{run}: reached {totals[run]} of {total_missed} known-bug edges the seeds miss "
            f"({totals[run] / max(1, total_missed):.1%}), out of {gained:,} new edges in total"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=Path("out/known_bugs"))
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("prepare", help="Find known bugs that crash the engine")
    p.add_argument("--v8", type=Path, default=Path("engines/v8/v8"))
    p.add_argument("--since", required=True, help="e.g. the engine version's branch date")
    p.add_argument("--until", required=True)
    p.add_argument("--engine-executable", type=Path, default=Path("engines/v8/v8/out/fuzzbuild/d8"))
    p.add_argument("--engine-mode", choices=["auto", "reprl", "remote"], default="auto")
    p.add_argument("--engine-address", default="localhost:9000")
    p.add_argument("--timeout", type=float, default=5.0, help="Longer than fuzzing (tests can be slow)")

    e = sub.add_parser("evaluate", help="Compare runs against the known bugs")
    e.add_argument("runs", type=Path, nargs="+")

    args = parser.parse_args()
    prepare(args) if args.command == "prepare" else evaluate(args)


if __name__ == "__main__":
    main()
