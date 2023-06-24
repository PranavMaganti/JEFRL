import random

import numpy as np
import torch


def setup_seeds() -> int:
    seed = random.randint(0, 2**32 - 1)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    return seed
