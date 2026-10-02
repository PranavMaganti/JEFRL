"""Sanity checks for an engine build before a long fuzzing run: REPRL works, coverage
is recorded, exceptions are not reported as crashes, and real crashes are."""

import argparse
from pathlib import Path
import sys
import time

from utils.js_engine import Engine
from utils.js_engine import JSError
from utils.js_engine import RemoteEngine
from utils.js_engine import ReprlEngine
from utils.js_engine import TIMEOUT


parser = argparse.ArgumentParser()
parser.add_argument("--engine-name", type=str, default="v8")
parser.add_argument(
    "--engine-executable", type=Path, default=Path("engines/v8/v8/out/fuzzbuild/d8")
)
parser.add_argument(
    "--engine-mode", choices=["auto", "reprl", "subprocess", "remote"], default="auto"
)
parser.add_argument("--engine-arg", action="append", default=[])
parser.add_argument("--timeout", type=float, default=TIMEOUT)
parser.add_argument(
    "--engine-address",
    default="localhost:9000",
    help="host:port of an engine server (src/utils/engine_server.py) for --engine-mode remote",
)
args = parser.parse_args()

failures = 0


def check(name: str, ok: bool, detail: str = ""):
    global failures
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f": {detail}" if detail else ""))
    failures += not ok


engine = Engine.get_engine(
    args.engine_name, args.engine_executable, args.engine_mode, args.timeout, args.engine_arg,
    address=args.engine_address,
)
# A remote engine server always uses REPRL
check("REPRL", isinstance(engine, (ReprlEngine, RemoteEngine)), type(engine).__name__)

data = engine.execute_text("var a = [1, 2, 3]; a.push(4); a.length;")
check(
    "coverage is recorded",
    data is not None and data.coverage.hit_edges > 0,
    f"{data.coverage.hit_edges} of {data.coverage.num_edges} edges" if data else "no result",
)
check("clean program has no error", data is not None and data.error == JSError.NoError)

data = engine.execute_text("undefinedFunction();")
check(
    "uncaught exception is not a crash",
    data is not None and not data.is_crash() and data.error != JSError.NoError,
    data.error.value if data else "no result",
)

data = engine.execute_text(
    "function f(x) { return x + 1; }\n"
    "%PrepareFunctionForOptimization(f); f(1); %OptimizeFunctionOnNextCall(f); f(2);"
)
check(
    "natives syntax is enabled",
    data is not None and data.error == JSError.NoError,
    (data.out + data.err).strip()[:200] if data else "no result",
)

# d8 built with v8_fuzzilli exposes fuzzilli('FUZZILLI_CRASH', n) to test crash
# detection: 0 = immediate crash, 1 = CHECK failure, 2 = DCHECK failure
for crash_type in range(3):
    data = engine.execute_text(f"fuzzilli('FUZZILLI_CRASH', {crash_type});")
    check(
        f"FUZZILLI_CRASH {crash_type} is detected as a crash",
        data is not None and data.is_crash(),
        (data.err.strip().splitlines() or [data.error.value])[-1][:200] if data else "no result",
    )

data = engine.execute_text("var a = 1;")
check("engine recovers after a crash", data is not None and data.error == JSError.NoError)

start = time.perf_counter()
n = 200
for i in range(n):
    engine.execute_text(f"var a = [{i}]; for (var i = 0; i < 100; i++) a.push(i);")
rate = n / (time.perf_counter() - start)
print(f"[INFO] throughput on a trivial program: {rate:.0f} execs/s per worker")

engine.close()
sys.exit(1 if failures else 0)
