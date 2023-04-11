from abc import ABC, abstractmethod
import ctypes
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from multiprocessing import shared_memory
from typing import Optional

from js_ast.nodes import Node
from js_ast import escodegen

SHM_SIZE = 0x100000
MAX_EDGES = (SHM_SIZE - 4) * 8
SHM_ID = "js_rl"

ENGINES_DIR = Path("engines")
CORPUS_DIR = Path("corpus")


class ShmData(ctypes.Structure):
    _fields_ = [
        ("num_edges", ctypes.c_uint32),
        ("edges", ctypes.c_ubyte * (SHM_SIZE - 4)),
    ]


class ExecutionData:
    def __init__(self, return_code=0, num_edges=0, hit_edges=0):
        self.return_code = return_code
        self.num_edges = num_edges
        self.hit_edges = hit_edges

    def coverage(self):
        return self.hit_edges / self.num_edges if self.num_edges > 0 else 0

    def is_crash(self):
        return self.return_code != 0


class Engine(ABC):
    def __init__(self) -> None:
        with open(self.get_corpus_lib(), "r") as f:
            self.lib = f.read()

    @abstractmethod
    def get_executable(self) -> str:
        pass

    @abstractmethod
    def get_corpus(self) -> str:
        pass

    @abstractmethod
    def get_corpus_lib(self) -> str:
        pass

    def execute_test(self, code: Node) -> Optional[ExecutionData]:
        # Write the code to a temporary file and execute it
        tmp = tempfile.NamedTemporaryFile(delete=True)
        tmp.write(self.lib.encode("utf-8"))
        tmp.write(escodegen.generate(code).encode("utf-8"))
        tmp.flush()

        shm = shared_memory.SharedMemory(name=SHM_ID, create=True, size=SHM_SIZE)
        os.environ["SHM_ID"] = SHM_ID

        try:
            popen = subprocess.Popen(
                [self.get_executable(), tmp.name], stdout=sys.stdout
            )
            popen.wait()
            
            data = ShmData.from_buffer(shm.buf)
            hit_edges = 0
            for i in range(math.ceil(data.num_edges / 8)):
                hit_edges += data.edges[i].bit_count()

            exec_data = ExecutionData(popen.returncode, data.num_edges, hit_edges)

            del data
            return exec_data

        except Exception as e:
            print(e)
        finally:
            shm.close()
            shm.unlink()
            tmp.close()


class V8Engine(Engine):
    def get_executable(self) -> str:
        return str(ENGINES_DIR / "v8/v8/out/fuzzbuild/d8")

    def get_corpus(self) -> str:
        return str(CORPUS_DIR / "v8")

    def get_corpus_lib(self) -> str:
        return str(CORPUS_DIR / "libs/v8.js")
