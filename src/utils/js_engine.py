import ctypes
import math
import os
import subprocess
import sys
import tempfile
from multiprocessing import shared_memory
from typing import Optional

from nodes.main import Node
from utils import escodegen

SHM_SIZE = 0x100000
MAX_EDGES = (SHM_SIZE - 4) * 8
SHM_ID = "test"


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


def execute_test(code: Node) -> Optional[ExecutionData]:
    tmp = tempfile.NamedTemporaryFile(delete=True)
    tmp.write(escodegen.generate(code).encode("utf-8"))
    shm = shared_memory.SharedMemory(name=SHM_ID, create=True, size=SHM_SIZE)
    os.environ["SHM_ID"] = SHM_ID

    try:
        popen = subprocess.Popen(
            ["./engines/v8/v8/out/fuzzbuild/d8", tmp.name], stdout=sys.stdout
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
