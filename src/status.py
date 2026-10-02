"""Summarises a fuzzing run from its stats.jsonl.

    python3 src/status.py out/rl-training/<run>      # a specific run
    python3 src/status.py                             # the latest run
"""

import argparse
import json
from pathlib import Path
import sys


parser = argparse.ArgumentParser()
parser.add_argument("run", type=Path, nargs="?", help="Run folder (default: latest)")
parser.add_argument(
    "--window", type=float, default=10, help="Minutes to compute recent rates over"
)
args = parser.parse_args()

run = args.run
if run is None:
    runs = sorted(Path("out/rl-training").glob("*/training/stats.jsonl"))
    if not runs:
        sys.exit("No runs found in out/rl-training")
    run = runs[-1].parent.parent

lines = [json.loads(line) for line in (run / "training" / "stats.jsonl").read_text().splitlines() if line]
if not lines:
    sys.exit(f"No stats yet in {run}")

now = lines[-1]
# Stats from `window` minutes ago, for recent rates
past = next(
    (s for s in reversed(lines) if now["elapsed_s"] - s["elapsed_s"] >= args.window * 60),
    lines[0],
)
span = max(1e-9, now["elapsed_s"] - past["elapsed_s"])


def hours(seconds: float) -> str:
    return f"{int(seconds // 3600)}h{int(seconds % 3600 // 60):02d}m"


def percent(part: float, total: float) -> str:
    return f"{100 * part / total:.0f}%" if total else "-"


crash_files = sorted((run / "interesting").glob("*_crash.js"))
checkpoint = run / "training" / "checkpoint" / "COMPLETE"

print(f"Run {run} ({hours(now['elapsed_s'])} elapsed)")
print()
print(f"  Coverage     {now['coverage']:.4%}  (+{now['coverage_gain'] * 100:.3f} points since start, "
      f"+{(now['coverage'] - past['coverage']) * 100:.4f} in the last {hours(span)})")
print(f"  Executions   {now['executions']:,}  ({now['execs_per_s']:.1f}/s overall, "
      f"{(now['executions'] - past['executions']) / span:.1f}/s recently, "
      f"{now.get('engine_utilisation', 0):.1f} engines busy)")
print(f"  Corpus       {now['corpus_size']:,}  (+{now['corpus_added']:,} added, -{now['corpus_removed']:,} removed; "
      f"+{now['corpus_added'] - past['corpus_added']} recently)")
print(f"  Crashes      {now['crashes']}" + (f"  ->  {', '.join(p.name for p in crash_files[:5])}" if crash_files else ""))
print(f"  Step errors  {now.get('step_errors', 0)}")
print()

errors = now["exec_errors"]
total = sum(errors.values())
recent = {k: errors.get(k, 0) - past["exec_errors"].get(k, 0) for k in errors}
recent_total = sum(recent.values())
print(f"  Executions   clean {percent(errors.get('NoError', 0), total)}, "
      f"timeout {percent(errors.get('TimeoutError', 0), total)}, "
      f"exception {percent(total - errors.get('NoError', 0) - errors.get('TimeoutError', 0), total)}"
      f"   (recently: clean {percent(recent.get('NoError', 0), recent_total)}, "
      f"timeout {percent(recent.get('TimeoutError', 0), recent_total)})")
if "rollbacks" in now:
    print(f"  Rollbacks    {now['rollbacks']:,} mutations undone for breaking the program or timing out "
          f"({percent(now['rollbacks'], total)} of executions; "
          f"recently {percent(now['rollbacks'] - past.get('rollbacks', 0), recent_total)})")

actions = now["actions"]
total_actions = sum(actions.values())
print("  Actions      " + ", ".join(
    f"{name} {percent(count, total_actions)}"
    for name, count in sorted(actions.items(), key=lambda kv: -kv[1])
))
print()

if "actor_time_s" in now and "actor_time_s" in past:
    # How the workers spent the recent window (summed over workers)
    hyperparameters = json.loads((run / "hyperparameters.json").read_text())
    workers = (hyperparameters.get("resumes") or [hyperparameters])[-1]["workers"]
    budget = span * workers
    delta = lambda k: now["actor_time_s"][k] - past["actor_time_s"][k]
    exec_s = now["time_exec_s"] - past["time_exec_s"]
    timeout_s = now.get("time_timeouts_s", 0) - past.get("time_timeouts_s", 0)
    if min(delta("env_step"), exec_s) >= 0:
        print(
            f"  Worker time  executing {percent(exec_s - timeout_s, budget)} "
            f"(+ timeouts {percent(timeout_s, budget)}), "
            f"mutating/other env work {percent(delta('env_step') - exec_s, budget)}, "
            f"GPU {percent(delta('gpu'), budget)}, waiting for GPU lock {percent(delta('lock_wait'), budget)}"
        )
        print()

if now.get("epsilon") is not None:
    print(f"  Agent        epsilon {now['epsilon']:.3f}, mean episode reward {now.get('mean_return_last_100') or 0:.2f} "
          f"(was {past.get('mean_return_last_100') or 0:.2f}), loss {now.get('mean_loss_last_1000') or 0:.3f}")
if checkpoint.exists():
    info = json.loads(checkpoint.read_text())
    print(f"  Checkpoint   at step {info['steps']:,}")
