"""Saving and loading fuzzing-run checkpoints, so a run can be stopped and resumed.

A checkpoint is a folder with the training state (networks, optimiser, replay
buffer, counters) in training.pt and the fuzzing state (corpus, coverage) in
corpus.pkl. It is written to a new folder and then swapped in, so an interrupted
save never damages the previous checkpoint.
"""

import json
import os
from pathlib import Path
import pickle
import shutil
import time
from typing import Any, Optional

import torch


CHECKPOINT = "checkpoint"
_NEW = "checkpoint-new"
_OLD = "checkpoint-old"
_COMPLETE = "COMPLETE"


def save_checkpoint(
    folder: Path,
    training_state: dict[str, Any],
    corpus_state: dict[str, Any],
    info: dict[str, Any],
):
    new, current, old = folder / _NEW, folder / CHECKPOINT, folder / _OLD
    shutil.rmtree(new, ignore_errors=True)
    new.mkdir(parents=True)

    torch.save(training_state, new / "training.pt")
    with open(new / "corpus.pkl", "wb") as f:
        pickle.dump(corpus_state, f, protocol=pickle.HIGHEST_PROTOCOL)
    # Written last: a checkpoint without it is incomplete and is ignored
    (new / _COMPLETE).write_text(json.dumps({**info, "saved_at": time.time()}))

    shutil.rmtree(old, ignore_errors=True)
    if current.exists():
        os.replace(current, old)
    os.replace(new, current)
    shutil.rmtree(old, ignore_errors=True)


def find_checkpoint(folder: Path) -> Optional[Path]:
    """The latest complete checkpoint in a run folder. Falls back to the previous
    one if the process died while swapping them."""
    for name in (CHECKPOINT, _OLD):
        path = folder / name
        if (path / _COMPLETE).exists():
            return path
    return None


def load_checkpoint(path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    training_state = torch.load(path / "training.pt", map_location="cpu", weights_only=False)
    with open(path / "corpus.pkl", "rb") as f:
        corpus_state = pickle.load(f)
    info = json.loads((path / _COMPLETE).read_text())
    return training_state, corpus_state, info
