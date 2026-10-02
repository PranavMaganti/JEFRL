from __future__ import annotations

from abc import ABC, abstractmethod
import ctypes
from dataclasses import dataclass
from enum import Enum
import fcntl
import itertools
import logging
import math
from multiprocessing import shared_memory
import os
from pathlib import Path
import resource
import select
import signal
import subprocess
import tempfile
from typing import Any, Optional

import numpy as np
from numpy.typing import NDArray


SHM_SIZE = 0x100000
MAX_EDGES = (SHM_SIZE - 4) * 8

TIMEOUT = 0.5

ENGINES_DIR = Path("engines")
CORPUS_DIR = Path("corpus")

# Make sanitizer failures abort (so they show up as a signal) and don't treat
# memory leaks at exit as crashes. These are ignored by non-sanitizer builds.
SANITIZER_ENV = {
    "ASAN_OPTIONS": "detect_leaks=0:abort_on_error=1:symbolize=0:allocator_may_return_null=1",
    "UBSAN_OPTIONS": "halt_on_error=1:abort_on_error=1:print_stacktrace=1",
}

# Output which means the engine hit a real bug (as opposed to a JS exception)
CRASH_MARKERS = (
    "AddressSanitizer",
    "UndefinedBehaviorSanitizer",
    "runtime error:",
    "# Fatal error in",
    "Check failed:",
    "Debug check failed:",
    "Safely terminating process due to CHECK failure",
)

# Out-of-memory aborts are expected when fuzzing and are not bugs
OOM_MARKERS = (
    "Fatal process out of memory",
    "Fatal JavaScript out of memory",
    "Fatal javascript OOM",
    "Fatal JavaScript invalid size error",
    "Fatal JavaScript invalid array length",
    "out of memory: Cannot allocate",
)


class JSError(str, Enum):
    ReferenceError = "ReferenceError"
    SyntaxError = "SyntaxError"
    TypeError = "TypeError"
    RangeError = "RangeError"
    URIError = "URIError"
    # Any other uncaught exception (or one whose message we could not read)
    Other = "Other"
    OutOfMemory = "OutOfMemory"
    TimeoutError = "TimeoutError"
    NoError = "NoError"


class ShmData(ctypes.Structure):
    _fields_ = [
        ("num_edges", ctypes.c_uint32),
        ("edges", ctypes.c_ubyte * (SHM_SIZE - 4)),
    ]


class Coverage:
    __slots__ = ["num_edges", "edges", "hit_edges"]

    def __init__(self, num_edges: int = 0, edges: Optional[NDArray[np.uint8]] = None):
        self.num_edges = num_edges
        self.edges = (
            np.zeros(math.ceil(self.num_edges / 8), dtype=np.uint8)
            if edges is None
            else edges
        )
        self.hit_edges: int = np.unpackbits(self.edges).sum()  # type: ignore

    def coverage(self) -> float:
        return self.hit_edges / self.num_edges if self.num_edges > 0 else 0

    def __or__(self, __value: Any) -> Coverage:
        if not isinstance(__value, Coverage):
            raise TypeError(
                "Cannot perform bitwise or on CoverageData and " + type(__value)
            )
        if self.num_edges == __value.num_edges:
            return Coverage(
                self.num_edges,
                self.edges | __value.edges,
            )
        elif self.num_edges == 0:
            return __value
        elif __value.num_edges == 0:
            return self

        raise ValueError("Cannot perform bitwise or on CoverageData with given objects")

    def __and__(self, __value: object):
        if not isinstance(__value, Coverage):
            raise TypeError(
                f"Cannot perform bitwise and on CoverageData and {type(__value)}"
            )

        if self.num_edges == __value.num_edges:
            return Coverage(
                self.num_edges,
                self.edges & __value.edges,
            )
        elif self.num_edges == 0:
            return __value
        elif __value.num_edges == 0:
            return self

        raise ValueError(
            "Cannot perform bitwise and on CoverageData with given objects"
        )

    def __str__(self) -> str:
        return f"{self.coverage():.5%}"

    def __eq__(self, __value: object) -> bool:
        if not isinstance(__value, Coverage):
            return False

        if self.num_edges != __value.num_edges:
            return False

        return (
            self.edges == __value.edges
        ).all() and self.num_edges == __value.num_edges

    def __deepcopy__(self, memo: dict[int, Any]) -> Coverage:
        return Coverage(self.num_edges, self.edges.copy())


@dataclass(slots=True)
class ExecutionData:
    coverage: Coverage
    error: JSError
    out: str
    err: str = ""
    crashed: bool = False

    def is_crash(self):
        return self.crashed


def classify_error(output: str, failed: bool) -> JSError:
    if any(marker in output for marker in OOM_MARKERS):
        return JSError.OutOfMemory
    if not failed:
        return JSError.NoError

    for error in (
        JSError.ReferenceError,
        JSError.SyntaxError,
        JSError.TypeError,
        JSError.RangeError,
        JSError.URIError,
    ):
        if error.value in output:
            return error

    return JSError.Other


def is_crash(signal_number: Optional[int], stderr: str) -> bool:
    """Decides whether an execution hit an engine bug.

    A JS exception only makes d8 exit with a non-zero status, whereas engine bugs
    kill it with a signal (segfault, CHECK/DCHECK trap, sanitizer abort) and/or
    print a sanitizer or V8 fatal error report.
    """
    if any(marker in stderr for marker in OOM_MARKERS):
        return False
    if signal_number is not None and signal_number != signal.SIGKILL:
        return True
    return any(marker in stderr for marker in CRASH_MARKERS)


class CoverageMap:
    """Shared memory region the instrumented engine writes edge coverage into
    (see src/fuzzilli/cov.cc in V8)."""

    _ids = itertools.count()

    def __init__(self):
        # Short name as macOS limits shared memory names to 31 characters
        self.name = f"jefrl_{os.getpid()}_{next(CoverageMap._ids)}"
        self.shm = shared_memory.SharedMemory(
            name=self.name, create=True, size=SHM_SIZE
        )
        self._data: Optional[ShmData] = ShmData.from_buffer(self.shm.buf)

    def clear(self):
        assert self._data is not None
        num_edges = int(self._data.num_edges)
        num_bytes = math.ceil(num_edges / 8) if num_edges > 0 else SHM_SIZE - 4
        ctypes.memset(ctypes.addressof(self._data.edges), 0, num_bytes)

    def read(self) -> Coverage:
        assert self._data is not None
        num_edges = int(self._data.num_edges)
        num_bytes = math.ceil(num_edges / 8)
        edges = np.frombuffer(self.shm.buf, dtype=np.uint8, count=num_bytes, offset=4)
        return Coverage(num_edges, edges.copy())

    def close(self):
        if self._data is None:
            return
        # The ctypes view must be released before the shared memory can be closed
        self._data = None
        self.shm.close()
        try:
            self.shm.unlink()
        except FileNotFoundError:
            pass


def disable_core_dumps():
    """Stops crashing engines from dumping core (engine processes inherit the
    limit). Dumping a large engine binary takes seconds, which turns the crash
    into a timeout, and fills the disk. On systemd systems (core_pattern piped
    to systemd-coredump) dumps are on by default."""
    _, hard = resource.getrlimit(resource.RLIMIT_CORE)
    resource.setrlimit(resource.RLIMIT_CORE, (0, hard))


class Engine(ABC):
    def __init__(
        self,
        executable: Path,
        timeout: float = TIMEOUT,
        extra_args: Optional[list[str]] = None,
    ) -> None:
        if not executable.exists():
            raise FileNotFoundError(f"Executable not found: {executable}")

        self.executable = executable.absolute()
        self.timeout = timeout
        disable_core_dumps()
        self.extra_args = extra_args or []
        self.coverage_map = CoverageMap()

    @staticmethod
    def get_engine(
        name: str,
        executable: Path,
        mode: str = "auto",
        timeout: float = TIMEOUT,
        extra_args: Optional[list[str]] = None,
        address: Optional[str] = None,
    ) -> Engine:
        """Creates an engine. `mode` is "reprl" (persistent process using Fuzzilli's
        read-eval-print-loop protocol), "subprocess" (a new process per execution),
        "auto" (REPRL if the binary supports it, otherwise subprocess) or "remote"
        (an engine server at `address`, see utils/engine_server.py)."""
        if mode == "remote":
            if address is None:
                raise ValueError("remote engines need an address (host:port)")
            return RemoteEngine(address)

        match name:
            case "v8":
                if mode in ("auto", "reprl"):
                    try:
                        return V8ReprlEngine(executable, timeout, extra_args)
                    except ReprlError as e:
                        if mode == "reprl":
                            raise
                        logging.warning(
                            f"REPRL unavailable ({e}), falling back to one process per execution"
                        )
                return V8Engine(executable, timeout, extra_args)

        raise ValueError(f"Unknown engine: {name}")

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def args(self) -> list[str]:
        pass

    def child_env(self) -> dict[str, str]:
        env = dict(os.environ)
        env.update(SANITIZER_ENV)
        env["SHM_ID"] = self.coverage_map.name
        return env

    def execute_text(
        self, code: str, lib_path: Optional[Path] = None
    ) -> Optional[ExecutionData]:
        if lib_path:
            code = f"load('{lib_path}');\n" + code

        return self.execute_script(code.encode("utf-8"))

    @abstractmethod
    def execute_script(self, script: bytes) -> Optional[ExecutionData]:
        pass

    def close(self):
        self.coverage_map.close()

    def __enter__(self) -> Engine:
        return self

    def __exit__(self, *_: Any):
        self.close()


class SubprocessEngine(Engine):
    """Runs every input in a fresh engine process. Simple but slow, as the engine
    start-up cost is paid on every execution."""

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        fd, path = tempfile.mkstemp(prefix="jefrl_", suffix=".js")
        os.close(fd)
        self.script_path = Path(path)

    def execute_script(self, script: bytes) -> Optional[ExecutionData]:
        self.script_path.write_bytes(script)
        self.coverage_map.clear()

        try:
            res = subprocess.run(
                [self.executable, *self.args, self.script_path],
                capture_output=True,
                check=False,
                timeout=self.timeout,
                env=self.child_env(),
            )
        except subprocess.TimeoutExpired as e:
            logging.debug("Timeout")
            return ExecutionData(
                self.coverage_map.read(),
                JSError.TimeoutError,
                (e.stdout or b"").decode("utf-8", errors="replace"),
                (e.stderr or b"").decode("utf-8", errors="replace"),
            )

        out = res.stdout.decode("utf-8", errors="replace")
        err = res.stderr.decode("utf-8", errors="replace")
        signal_number = -res.returncode if res.returncode < 0 else None

        if "Check failed: read(100" in err:
            # d8 8.5 built with v8_fuzzilli=true always expects the REPRL channel,
            # so every run would otherwise look like a crash
            raise RuntimeError(
                f"{self.executable} only supports REPRL mode (built with v8_fuzzilli=true)"
            )

        return ExecutionData(
            self.coverage_map.read(),
            classify_error(out + err, res.returncode != 0),
            out,
            err,
            is_crash(signal_number, err),
        )

    def close(self):
        self.script_path.unlink(missing_ok=True)
        super().close()


class ReprlError(Exception):
    pass


class ReprlEngine(Engine):
    """Keeps one engine process alive and sends it inputs over Fuzzilli's REPRL
    (read-eval-print-loop) protocol, avoiding the process start-up cost on every
    execution. d8 supports this when built with `v8_fuzzilli=true`.

    Protocol (see Fuzzilli's libreprl and V8's src/d8/d8.cc):
      fd 100 (CRFD): parent -> child control: "exec" + u64 script length
      fd 101 (CWFD): child -> parent control: "HELO" handshake, then u32 status
      fd 102 (DRFD): script source, read by the child from offset 0
      fd 103 (DWFD): child -> parent data (unused by d8)
    """

    CRFD, CWFD, DRFD, DWFD = 100, 101, 102, 103
    HANDSHAKE_TIMEOUT = 30.0
    # Restart the engine periodically so state doesn't build up across executions
    MAX_EXECS_PER_PROCESS = 1000

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.pid: Optional[int] = None
        self.execs_since_spawn = 0
        self._fds: list[int] = []
        try:
            self._spawn()
        except Exception:
            self.close()
            raise

    def _spawn(self):
        ctrl_in_r, ctrl_in_w = os.pipe()  # parent -> child
        ctrl_out_r, ctrl_out_w = os.pipe()  # child -> parent

        # Regular (unlinked) files so the parent can rewind/truncate them. The child
        # shares the open file description, so rewinding here rewinds the child too.
        data_fd = self._unlinked_file(os.O_RDWR)
        stdout_fd = self._unlinked_file(os.O_RDWR | os.O_APPEND)
        stderr_fd = self._unlinked_file(os.O_RDWR | os.O_APPEND)
        devnull_r = os.open(os.devnull, os.O_RDONLY)
        devnull_w = os.open(os.devnull, os.O_WRONLY)

        child_fds = {
            0: devnull_r,
            1: stdout_fd,
            2: stderr_fd,
            self.CRFD: ctrl_in_r,
            self.CWFD: ctrl_out_w,
            self.DRFD: data_fd,
            self.DWFD: devnull_w,
        }
        # Duplicate onto high numbers first so no source fd can be clobbered by an
        # earlier dup2 onto one of the target numbers
        high_fds = {
            target: fcntl.fcntl(fd, fcntl.F_DUPFD_CLOEXEC, 256)
            for target, fd in child_fds.items()
        }
        file_actions = [
            (os.POSIX_SPAWN_DUP2, fd, target) for target, fd in high_fds.items()
        ]

        try:
            self.pid = os.posix_spawn(
                str(self.executable),
                [str(self.executable), *self.args],
                self.child_env(),
                file_actions=file_actions,
            )
        finally:
            for fd in [*high_fds.values(), ctrl_in_r, ctrl_out_w, devnull_r, devnull_w]:
                os.close(fd)

        self.ctrl_in_w = ctrl_in_w
        self.ctrl_out_r = ctrl_out_r
        self.data_fd = data_fd
        self.stdout_fd = stdout_fd
        self.stderr_fd = stderr_fd
        self._fds = [ctrl_in_w, ctrl_out_r, data_fd, stdout_fd, stderr_fd]
        self.execs_since_spawn = 0

        helo = self._read_ctrl(self.HANDSHAKE_TIMEOUT)
        if helo != b"HELO":
            stderr = self._drain(self.stderr_fd)
            self._kill()
            raise ReprlError(
                f"no REPRL handshake from {self.executable.name} "
                f"(is it built with v8_fuzzilli=true?): {stderr.strip()[:500]}"
            )
        os.write(self.ctrl_in_w, b"HELO")

    @staticmethod
    def _unlinked_file(flags: int) -> int:
        fd, path = tempfile.mkstemp(prefix="jefrl_reprl_")
        os.close(fd)
        fd = os.open(path, flags)
        os.unlink(path)
        return fd

    def _read_ctrl(self, timeout: float) -> Optional[bytes]:
        """Returns 4 bytes from the child, b"" if it died, None on timeout."""
        ready, _, _ = select.select([self.ctrl_out_r], [], [], timeout)
        if not ready:
            return None

        data = b""
        while len(data) < 4:
            chunk = os.read(self.ctrl_out_r, 4 - len(data))
            if not chunk:
                return b""
            data += chunk
        return data

    @staticmethod
    def _drain(fd: int) -> str:
        size = os.fstat(fd).st_size
        data = os.pread(fd, size, 0) if size else b""
        os.ftruncate(fd, 0)
        return data.decode("utf-8", errors="replace")

    def _kill(self) -> Optional[int]:
        """Kills and reaps the child. Returns the signal that killed it (if any)."""
        if self.pid is None:
            return None

        pid, self.pid = self.pid, None
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        return self._reap(pid)

    @staticmethod
    def _reap(pid: int) -> Optional[int]:
        try:
            _, status = os.waitpid(pid, 0)
        except ChildProcessError:
            return None
        return os.WTERMSIG(status) if os.WIFSIGNALED(status) else None

    def _child_died(self) -> Optional[int]:
        """Reaps a child that exited by itself. Returns the signal that killed it."""
        pid, self.pid = self.pid, None
        return self._reap(pid) if pid is not None else None

    def _close_fds(self):
        for fd in self._fds:
            try:
                os.close(fd)
            except OSError:
                pass
        self._fds = []

    def _restart(self):
        self._kill()
        self._close_fds()
        self._spawn()

    def execute_script(self, script: bytes) -> Optional[ExecutionData]:
        if self.pid is None or self.execs_since_spawn >= self.MAX_EXECS_PER_PROCESS:
            self._restart()

        self.coverage_map.clear()
        os.ftruncate(self.data_fd, 0)
        os.pwrite(self.data_fd, script, 0)
        os.lseek(self.data_fd, 0, os.SEEK_SET)

        try:
            os.write(self.ctrl_in_w, b"exec" + len(script).to_bytes(8, "little"))
        except BrokenPipeError:
            # The process died between executions, start a new one and retry
            self._restart()
            return self.execute_script(script)

        self.execs_since_spawn += 1
        status = self._read_ctrl(self.timeout)

        if status is None:
            self._kill()
            out, err = self._drain(self.stdout_fd), self._drain(self.stderr_fd)
            coverage = self.coverage_map.read()
            self._close_fds()
            logging.debug("Timeout")
            # A crash can race with the timeout, keep it if the child reported one
            return ExecutionData(
                coverage,
                JSError.TimeoutError,
                out,
                err,
                is_crash(None, err),
            )

        if status == b"":
            # The engine died while executing the input
            signal_number = self._child_died()
            out, err = self._drain(self.stdout_fd), self._drain(self.stderr_fd)
            coverage = self.coverage_map.read()
            self._close_fds()
            return ExecutionData(
                coverage,
                classify_error(out + err, True),
                out,
                err,
                is_crash(signal_number, err),
            )

        exit_code = (int.from_bytes(status, "little", signed=True) >> 8) & 0xFF
        out, err = self._drain(self.stdout_fd), self._drain(self.stderr_fd)

        # d8's stdout is block buffered when writing to a file, so the exception
        # message may not be visible yet. The exit code is always accurate.
        return ExecutionData(
            self.coverage_map.read(),
            classify_error(out + err, exit_code != 0),
            out,
            err,
            is_crash(None, err),
        )

    def close(self):
        self._kill()
        self._close_fds()
        super().close()


V8_ARGS = [
    "--expose-gc",
    "--omit-quit",
    "--allow-natives-syntax",
    "--fuzzing",
    # "--jit-fuzzing",
    "--future",
    "--harmony",
]


class V8Engine(SubprocessEngine):
    @property
    def name(self) -> str:
        return "V8"

    @property
    def args(self) -> list[str]:
        return [*V8_ARGS, *self.extra_args]


class V8ReprlEngine(ReprlEngine):
    @property
    def name(self) -> str:
        return "V8"

    @property
    def args(self) -> list[str]:
        return [*V8_ARGS, *self.extra_args]


class RemoteEngine(Engine):
    """Executes programs on an engine server (utils/engine_server.py), e.g. one
    running in a Linux container while the fuzzer runs natively. The server owns
    the engine process, its flags and its timeout."""

    def __init__(self, address: str):
        import socket

        host, port = address.rsplit(":", 1)
        self.address = address
        self.executable = Path(address)
        self.timeout = TIMEOUT
        self.extra_args = []
        self.sock = socket.create_connection((host, int(port)))
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    @property
    def name(self) -> str:
        return f"remote({self.address})"

    @property
    def args(self) -> list[str]:
        return []

    def execute_text(
        self, code: str, lib_path: Optional[Path] = None
    ) -> Optional[ExecutionData]:
        from utils.engine_server import recv_message
        from utils.engine_server import send_message

        send_message(
            self.sock, {"lib": str(lib_path) if lib_path else None}, code.encode("utf-8")
        )
        header, edges = recv_message(self.sock)
        if header.get("none"):
            return None

        return ExecutionData(
            Coverage(header["num_edges"], np.frombuffer(edges, dtype=np.uint8).copy()),
            JSError(header["error"]),
            header["out"],
            header["err"],
            header["crashed"],
        )

    def execute_script(self, script: bytes) -> Optional[ExecutionData]:
        return self.execute_text(script.decode("utf-8", errors="replace"))

    def close(self):
        self.sock.close()
