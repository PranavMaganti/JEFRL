import torch


def get_device() -> torch.device:
    """CUDA if available, then Apple Silicon (MPS), then CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
