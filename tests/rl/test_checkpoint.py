from collections import Counter
from pathlib import Path

import torch

from rl.checkpoint import find_checkpoint
from rl.checkpoint import load_checkpoint
from rl.checkpoint import save_checkpoint
from rl.dqn import ReplayBuffer
from rl.env import SharedFuzzingState
from rl.fuzzing_action import FuzzingAction
from rl.program_state import ProgramState
from utils.js_engine import Coverage
from utils.loader import parse_code


N_ACTIONS = len(FuzzingAction)


def test_save_and_load(tmp_path: Path):
    save_checkpoint(tmp_path, {"steps": 1, "t": torch.ones(3)}, {"corpus": [1, 2]}, {"steps": 1})
    path = find_checkpoint(tmp_path)
    assert path is not None
    training, corpus, info = load_checkpoint(path)
    assert training["steps"] == 1 and torch.equal(training["t"], torch.ones(3))
    assert corpus == {"corpus": [1, 2]}
    assert info["steps"] == 1


def test_latest_checkpoint_replaces_previous(tmp_path: Path):
    save_checkpoint(tmp_path, {"steps": 1}, {}, {"steps": 1})
    save_checkpoint(tmp_path, {"steps": 2}, {}, {"steps": 2})
    training, _, _ = load_checkpoint(find_checkpoint(tmp_path))
    assert training["steps"] == 2
    assert sorted(p.name for p in tmp_path.iterdir()) == ["checkpoint"]


def test_incomplete_save_keeps_previous_checkpoint(tmp_path: Path):
    save_checkpoint(tmp_path, {"steps": 1}, {}, {"steps": 1})
    # Simulate dying part-way through writing the next checkpoint
    (tmp_path / "checkpoint-new").mkdir()
    (tmp_path / "checkpoint-new" / "training.pt").write_bytes(b"partial")
    training, _, _ = load_checkpoint(find_checkpoint(tmp_path))
    assert training["steps"] == 1


def test_falls_back_to_old_checkpoint_mid_swap(tmp_path: Path):
    save_checkpoint(tmp_path, {"steps": 1}, {}, {"steps": 1})
    # Died after moving the current checkpoint aside, before moving the new one in
    (tmp_path / "checkpoint").rename(tmp_path / "checkpoint-old")
    training, _, _ = load_checkpoint(find_checkpoint(tmp_path))
    assert training["steps"] == 1


def test_no_checkpoint(tmp_path: Path):
    assert find_checkpoint(tmp_path) is None


def test_replay_buffer_round_trip():
    cpu = torch.device("cpu")
    buffer = ReplayBuffer(5, 4, N_ACTIONS, cpu)
    for i in range(7):  # wraps around
        buffer.push(
            torch.full((1, 4), float(i)),
            torch.tensor([i % N_ACTIONS]),
            torch.tensor([float(i)]),
            torch.full((1, 4), float(i + 1)),
            torch.tensor([i % 2 == 0]),
            torch.ones(1, N_ACTIONS, dtype=torch.bool),
            torch.tensor([0.9]),
        )

    restored = ReplayBuffer(5, 4, N_ACTIONS, cpu)
    restored.load_state_dict(buffer.state_dict())

    assert (restored.pos, restored.size) == (buffer.pos, buffer.size)
    for name in ReplayBuffer._TENSORS:
        assert torch.equal(getattr(restored, name), getattr(buffer, name))


def test_replay_buffer_partially_filled():
    cpu = torch.device("cpu")
    buffer = ReplayBuffer(10, 4, N_ACTIONS, cpu)
    buffer.push(
        torch.ones(3, 4),
        torch.zeros(3, dtype=torch.long),
        torch.ones(3),
        torch.ones(3, 4),
        torch.zeros(3, dtype=torch.bool),
        torch.ones(3, N_ACTIONS, dtype=torch.bool),
        torch.full((3,), 0.9),
    )
    restored = ReplayBuffer(10, 4, N_ACTIONS, cpu)
    restored.load_state_dict(buffer.state_dict())
    assert (restored.pos, restored.size) == (3, 3)


def test_corpus_snapshot_round_trip():
    ast = parse_code("var v0 = 1;", "seed.js")
    state = ProgramState(ast, None)  # type: ignore
    shared = SharedFuzzingState([state], Coverage(16), min_corpus_size=1)
    shared.entries[0].selections_without_increase = 7
    shared.num_crashes, shared.num_added, shared.num_removed = 1, 2, 3

    snapshot = shared.snapshot()
    restored = SharedFuzzingState([], Coverage())
    restored.restore(snapshot)

    assert len(restored) == 1
    assert restored.entries[0].state.generate_program_code() == "var v0 = 1;"
    assert restored.entries[0].selections_without_increase == 7
    assert restored.total_coverage == shared.total_coverage
    assert (restored.num_crashes, restored.num_added, restored.num_removed) == (1, 2, 3)


def test_env_stats_are_added():
    from rl.env import FuzzingEnv

    env = FuzzingEnv.__new__(FuzzingEnv)
    for name in FuzzingEnv._STATS:
        setattr(env, name, Counter() if name in ("exec_errors", "action_counts") else 0)
    env.add_stats({"total_executions": 5, "exec_errors": Counter({"NoError": 3}), "exec_time": 1.5})
    env.add_stats({"total_executions": 2, "exec_errors": Counter({"NoError": 1, "Other": 1})})
    assert env.total_executions == 7
    assert env.exec_errors == Counter({"NoError": 4, "Other": 1})
    assert env.exec_time == 1.5
