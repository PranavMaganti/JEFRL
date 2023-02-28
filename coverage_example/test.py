from multiprocessing import shared_memory, resource_tracker
import os
import subprocess
import sys
import ctypes
import math

SHM_SIZE = 0x100000
MAX_EDGES = (SHM_SIZE - 4) * 8
SHM_ID = "test"


class ShmData(ctypes.Structure):
    _fields_ = [
        ("num_edges", ctypes.c_uint32),
        ("edges", ctypes.c_ubyte * (SHM_SIZE - 4)),
    ]


shm = shared_memory.SharedMemory(name=SHM_ID, create=True, size=SHM_SIZE)
os.environ["SHM_ID"] = SHM_ID

try:
    popen = subprocess.Popen(["./engines/v8/v8/out/fuzzbuild/d8", "test.js"], stdout=sys.stdout)
    popen.wait()
    data = ShmData.from_buffer(shm.buf)

    print(f"Exited with code: {popen.returncode}")

    print("Total edges:", data.num_edges)

    hit_edges = 0
    for i in range(math.ceil(data.num_edges / 8)):
        hit_edges +=  data.edges[i].bit_count()
    
    print("Hit edges:", hit_edges)

    del data

except Exception as e:
    print(e)
finally:
    shm.close()
    shm.unlink()
