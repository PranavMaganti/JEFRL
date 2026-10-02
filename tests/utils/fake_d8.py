"""
A stand-in for an instrumented d8 used to test the engine wrappers without building
V8. It speaks Fuzzilli's REPRL protocol when fds 100-103 are open, otherwise it runs
the script file given as its last argument once.

Scripts are made of directives, one per line:
  COVER <n>   mark edge n as hit in the shared memory coverage map
  THROW       print an uncaught TypeError and exit with status 1
  CRASH       abort (SIGABRT), like a CHECK failure or sanitizer abort
  ASAN        print an AddressSanitizer report and exit with status 1
  OOM         print a V8 out-of-memory error and abort
  HANG        sleep forever
  PRINT <s>   print s to stdout
Other lines are ignored.
"""

from multiprocessing import resource_tracker
from multiprocessing import shared_memory
import os
import sys
import time
import zlib


NUM_EDGES = int(os.environ.get("FAKE_D8_EDGES", "64"))
CRFD, CWFD, DRFD = 100, 101, 102


def open_coverage():
    shm = shared_memory.SharedMemory(name=os.environ["SHM_ID"])
    # Attaching registers the segment with this process's resource tracker, which
    # would unlink it on exit. It belongs to the parent, so unregister it.
    resource_tracker.unregister(shm._name, "shared_memory")  # type: ignore
    shm.buf[0:4] = NUM_EDGES.to_bytes(4, "little")
    return shm


def run(script: str, shm) -> int:
    status = 0
    for line in script.splitlines():
        parts = line.strip().split(" ", 1)
        match parts[0]:
            case "COVER":
                edge = int(parts[1])
                shm.buf[4 + edge // 8] |= 1 << (edge % 8)
            case "THROW":
                print("TypeError: fake exception", flush=True)
                status = 1
                break
            case "CRASH":
                os.abort()
            case "ASAN":
                print("==1==ERROR: AddressSanitizer: heap-buffer-overflow", file=sys.stderr, flush=True)
                os._exit(1)
            case "OOM":
                print("# Fatal process out of memory: fake", file=sys.stderr, flush=True)
                os.abort()
            case "HANG":
                time.sleep(3600)
            case "PRINT":
                print(parts[1], flush=True)
            case _:
                # Any other line (e.g. real JS) hits a pseudo-random edge, so the
                # fuzzer sees coverage change as programs are mutated
                if line.strip() and not line.startswith("load("):
                    edge = zlib.crc32(line.strip().encode()) % NUM_EDGES
                    shm.buf[4 + edge // 8] |= 1 << (edge % 8)
    return status


def read_exact(fd: int, n: int) -> bytes:
    data = b""
    while len(data) < n:
        chunk = os.read(fd, n - len(data))
        if not chunk:
            sys.exit(0)
        data += chunk
    return data


def main():
    shm = open_coverage()

    reprl = "--no-reprl" not in sys.argv
    if reprl:
        try:
            os.write(CWFD, b"HELO")
            reprl = read_exact(CRFD, 4) == b"HELO"
        except OSError:
            reprl = False

    if not reprl:
        path = sys.argv[-1]
        if not os.path.isfile(path):
            sys.exit(0)
        with open(path) as f:
            sys.exit(run(f.read(), shm))

    while True:
        if read_exact(CRFD, 4) != b"exec":
            os._exit(255)
        size = int.from_bytes(read_exact(CRFD, 8), "little")
        script = read_exact(DRFD, size).decode()
        status = run(script, shm)
        os.write(CWFD, (status << 8).to_bytes(4, "little"))


if __name__ == "__main__":
    main()
