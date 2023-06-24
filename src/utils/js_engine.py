from __future__ import annotations

from abc import ABC, abstractmethod
import ctypes
from dataclasses import dataclass
from enum import Enum
import logging
import math
from multiprocessing import shared_memory
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Any, Optional

import numpy as np
from numpy.typing import NDArray


SHM_SIZE = 0x100000
MAX_EDGES = (SHM_SIZE - 4) * 8
SHM_ID = f"js_rl_{os.getpid()}"

TIMEOUT = 0.5

ENGINES_DIR = Path("engines")
CORPUS_DIR = Path("corpus")


class JSError(str, Enum):
    ReferenceError = "ReferenceError"
    SyntaxError = "SyntaxError"
    TypeError = "TypeError"
    RangeError = "RangeError"
    URIError = "URIError"
    Other = "Other"
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

    def is_crash(self):
        return self.error == JSError.Other


class Engine(ABC):
    def __init__(self, executable: Path) -> None:
        if not executable.exists():
            raise FileNotFoundError(f"Executable not found: {self.executable}")

        self.executable = executable

    @staticmethod
    def get_engine(name: str, executable: Path) -> Engine:
        match name:
            case "v8":
                return V8Engine(executable)

        raise ValueError(f"Unknown engine: {name}")

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def args(self) -> list[str]:
        pass

    def execute_text(
        self, code: str, lib_path: Optional[Path] = None
    ) -> Optional[ExecutionData]:
        tmp = tempfile.NamedTemporaryFile(delete=True)
        if lib_path:
            tmp.write(f"load('{lib_path}')\n".encode("utf-8"))
        tmp.write(code.encode("utf-8"))
        tmp.flush()

        return self.execute_file(tmp.name)

    def execute_file(self, file: str):
        shm = shared_memory.SharedMemory(name=SHM_ID, create=True, size=SHM_SIZE)
        os.environ["SHM_ID"] = SHM_ID
        out: Optional[str] = None

        try:
            res = subprocess.run(
                [self.executable, *self.args, file],
                capture_output=True,
                check=False,
                timeout=TIMEOUT,
            )
            try:
                out = res.stdout.decode("utf-8")
            except:
                shm.close()
                shm.unlink()

                return None

            error = JSError.NoError
            if "ReferenceError" in out:
                error = JSError.ReferenceError
            elif "SyntaxError" in out:
                error = JSError.SyntaxError
            elif "TypeError" in out:
                error = JSError.TypeError
            elif "RangeError" in out:
                error = JSError.RangeError
            elif "URIError" in out:
                error = JSError.URIError
            elif res.returncode != 0:
                error = JSError.Other
        except subprocess.TimeoutExpired:
            error = JSError.TimeoutError
            logging.debug("Timeout")

        data = ShmData.from_buffer(shm.buf)

        num_edges = int(data.num_edges)
        num_bytes = math.ceil(num_edges / 8)
        edges = np.array(np.ctypeslib.as_array(data.edges)[:num_bytes])

        exec_data = ExecutionData(
            Coverage(num_edges, edges),
            error,
            out if out is not None else "",
        )

        del data

        shm.close()
        shm.unlink()

        return exec_data


class V8Engine(Engine):
    @property
    def name(self) -> str:
        return "V8"

    @property
    def args(self) -> list[str]:
        return [
            "--expose-gc",
            "--omit-quit",
            "--allow-natives-syntax",
            "--fuzzing",
            # "--jit-fuzzing",
            "--future",
            "--harmony",
        ]
