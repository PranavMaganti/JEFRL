import copy
import hashlib
from pathlib import Path
import random
from typing import Optional

import numpy as np
from preprocessing.vocabulary import get_frag_counts
from preprocessing.vocabulary import get_frag_data
from preprocessing.vocabulary import get_vocab
import pytest
from rl.env import COVERAGE_REWARD
from rl.env import FuzzingEnv
from rl.env import SharedFuzzingState
from rl.fuzzing_action import FuzzingAction
from rl.program_state import ProgramState
from transformer.tokenizer import ASTTokenizer

from utils.js_engine import Coverage
from utils.js_engine import Engine
from utils.js_engine import ExecutionData
from utils.js_engine import JSError
from utils.loader import get_subtrees
from utils.loader import parse_code


NUM_EDGES = 256

SEEDS = [
    """
    var v0 = [1.1, 2.2, 3.3];
    function f0(v1) {
        var v2 = v1 + 1;
        return Math.floor(v2) * v0.length;
    }
    for (var v3 = 0; v3 < 100; v3++) {
        f0(v3);
    }
    %OptimizeFunctionOnNextCall(f0);
    f0(2);
    """,
    """
    class c0 {
        constructor(v0) { this.v1 = v0; }
        m() { return this.v1 - 1; }
    }
    let v2 = new c0(5);
    let v3 = v2.m() << 2;
    try { v3 = v3 / 0; } catch (e) { v3 = e; }
    """,
    """
    const v0 = new Array(10).fill(0).map((x, i) => x + i);
    var v1 = { a: 1, b: v0 };
    if (v1.a > 0) { v1.b[0] = !v1.a; } else { v1.a++; }
    """,
]


def fake_coverage(code: str) -> Coverage:
    # Each distinct line of code hits one pseudo-random edge
    edges = np.zeros(NUM_EDGES // 8, dtype=np.uint8)
    for line in code.splitlines():
        edge = int(hashlib.md5(line.strip().encode()).hexdigest(), 16) % NUM_EDGES
        edges[edge // 8] |= 1 << (edge % 8)
    return Coverage(NUM_EDGES, edges)


class FakeEngine(Engine):
    def __init__(self):
        self.executed: list[str] = []

    @property
    def name(self) -> str:
        return "fake"

    @property
    def args(self) -> list[str]:
        return []

    def execute_text(self, code: str, lib_path: Optional[Path] = None):
        self.executed.append(code)
        return ExecutionData(fake_coverage(code), JSError.NoError, "")

    def execute_script(self, script: bytes):
        raise NotImplementedError

    def close(self):
        pass


def make_env(**kwargs) -> tuple[FuzzingEnv, SharedFuzzingState, FakeEngine]:
    corpus: list[ProgramState] = []
    for code in SEEDS:
        ast = parse_code(code, "seed.js")
        assert ast is not None
        state = ProgramState(ast, None)  # type: ignore
        state.exec_data = ExecutionData(
            fake_coverage(state.generate_program_code()), JSError.NoError, ""
        )
        corpus.append(state)

    subtrees = get_subtrees(corpus)
    frag_seqs, node_types = get_frag_data(corpus)
    vocab, token_to_id, _, _ = get_vocab(get_frag_counts(frag_seqs), node_types, 1)
    tokenizer = ASTTokenizer(vocab, token_to_id, 512)

    total = Coverage()
    for state in corpus:
        total |= state.exec_data.coverage

    shared = SharedFuzzingState(copy.deepcopy(corpus), total, min_corpus_size=1)
    engine = FakeEngine()
    env = FuzzingEnv(shared, subtrees, engine, tokenizer, **kwargs)
    return env, shared, engine


@pytest.fixture(autouse=True)
def seed():
    random.seed(0)
    np.random.seed(0)


class TestActionMask:
    def test_mask_matches_action_success(self):
        """Every action allowed by the mask succeeds, every other action fails."""
        env, _, _ = make_env()
        env.reset(seed=0)
        rng = random.Random(0)
        checked = 0

        for _ in range(300):
            mask = env.action_mask()
            for action in FuzzingAction:
                if action == FuzzingAction.END:
                    continue
                state = copy.deepcopy(env._state)
                # deepcopy resets the target, so find the same node in the copy
                path = []
                node = env._state.target_node
                while node.parent is not None:
                    siblings = node.parent.children()
                    path.append(next(i for i, c in enumerate(siblings) if c is node))
                    node = node.parent
                target = state.root
                for i in reversed(path):
                    target = target.children()[i]
                state.target_node = target

                match action:
                    case FuzzingAction.REPLACE:
                        _, ok = state.replace(env.subtrees)
                    case FuzzingAction.ADD:
                        _, ok = state.add(env.subtrees)
                    case FuzzingAction.REMOVE:
                        _, ok = state.remove(env.subtrees)
                    case FuzzingAction.MODIFY:
                        ok = state.modify()
                    case FuzzingAction.MOVE_UP:
                        ok = state.move_up()
                    case FuzzingAction.MOVE_DOWN:
                        ok = state.move_down()
                    case FuzzingAction.MOVE_LEFT:
                        ok = state.move_left()
                    case FuzzingAction.MOVE_RIGHT:
                        ok = state.move_right()
                assert ok == mask[action], (action, env._state.target_node.type)
                checked += 1

            valid = [a for a in FuzzingAction if mask[a] and a != FuzzingAction.END]
            _, _, terminated, truncated, _ = env.step(rng.choice(valid))
            if terminated or truncated:
                env.reset()

        assert checked > 1000


class TestEpisodes:
    def test_step_cap_truncates(self):
        env, _, _ = make_env(max_steps=5)
        env.reset(seed=0)
        for i in range(5):
            action = (
                FuzzingAction.MOVE_DOWN
                if env.action_mask()[FuzzingAction.MOVE_DOWN]
                else FuzzingAction.MOVE_UP
            )
            _, reward, terminated, truncated, _ = env.step(action)
            assert not terminated
            if i < 4:
                assert reward == 0 and not truncated

        assert truncated
        # Ending an episode costs nothing: executions are what cost
        assert reward == 0

    def test_end_is_free(self):
        env, _, _ = make_env()
        env.reset(seed=0)
        _, reward, terminated, truncated, _ = env.step(FuzzingAction.END)
        assert terminated and not truncated
        assert reward == 0


class TestRewards:
    def test_only_new_total_coverage_is_rewarded(self):
        """A program increasing its own coverage earns nothing unless the run gains
        coverage, otherwise removing code and adding it back pays."""
        env, shared, engine = make_env()
        env.reset(seed=0)
        cost = shared.execution_cost()
        before = len(shared)

        def own_coverage_up(code, lib_path=None):
            # More edges than the program had, but all already in the total
            return ExecutionData(copy.deepcopy(shared.total_coverage), JSError.NoError, "")

        engine.execute_text = own_coverage_up
        reward = mutate_once(env)
        assert reward == pytest.approx(-cost)
        assert len(shared) == before

    def test_new_total_coverage_beats_the_cost(self):
        env, shared, engine = make_env()
        env.reset(seed=0)
        cost = shared.execution_cost()

        def everything(code, lib_path=None):
            edges = np.full(NUM_EDGES // 8, 0xFF, dtype=np.uint8)
            return ExecutionData(Coverage(NUM_EDGES, edges), JSError.NoError, "")

        engine.execute_text = everything
        assert mutate_once(env) == pytest.approx(COVERAGE_REWARD - cost)

    def test_cost_follows_the_rate_of_new_coverage(self):
        env, shared, engine = make_env()
        env.reset(seed=0)
        start = shared.coverage_rate
        engine.execute_text = lambda code, lib_path=None: ExecutionData(
            copy.deepcopy(shared.total_coverage), JSError.NoError, ""
        )
        for _ in range(20):
            mutate_once(env)
        # Nothing new was found, so the expected reward per execution fell
        assert shared.coverage_rate < start
        assert shared.execution_cost() == pytest.approx(COVERAGE_REWARD * shared.coverage_rate)

    def test_rate_is_checkpointed(self):
        _, shared, _ = make_env()
        shared.coverage_rate = 0.0123
        snapshot = shared.snapshot()
        _, restored, _ = make_env()
        restored.restore(snapshot)
        assert restored.coverage_rate == 0.0123

    def test_new_corpus_entry_is_not_mutated_afterwards(self):
        env, shared, engine = make_env()
        rng = random.Random(1)
        env.reset(seed=1)

        for _ in range(2000):
            before = len(shared)
            valid = [
                a
                for a in (FuzzingAction.REPLACE, FuzzingAction.ADD, FuzzingAction.MODIFY)
                if env.action_mask()[a]
            ] or [FuzzingAction.MOVE_UP]
            _, _, terminated, truncated, _ = env.step(rng.choice(valid))

            if len(shared) > before:
                added = shared.entries[-1].state
                code_when_added = engine.executed[-1]
                assert added.generate_program_code() == code_when_added
                assert added.exec_data.coverage == fake_coverage(code_when_added)
                # Keep mutating the episode's program...
                for _ in range(5):
                    if env.action_mask()[FuzzingAction.REPLACE]:
                        env.step(FuzzingAction.REPLACE)
                # ...the stored copy must not change
                assert added.generate_program_code() == code_when_added
                return

            if terminated or truncated:
                env.reset()

        pytest.fail("coverage never increased")


def throwing_with_lots_of_coverage(code, lib_path=None):
    edges = np.full(NUM_EDGES // 8, 0xFF, dtype=np.uint8)  # every edge "hit"
    return ExecutionData(Coverage(NUM_EDGES, edges), JSError.TypeError, "")


def mutate_once(env: FuzzingEnv) -> Optional[float]:
    """Applies one mutation (moving up until one is possible), returns its reward."""
    for _ in range(50):
        valid = [a for a in (FuzzingAction.REPLACE, FuzzingAction.ADD) if env.action_mask()[a]]
        if valid:
            return env.step(valid[0])[1]
        env.step(FuzzingAction.MOVE_UP)
    return None


class TestRollback:
    def test_mutation_which_makes_a_clean_program_throw_is_undone(self):
        env, shared, engine = make_env()
        env.reset(seed=0)
        engine.execute_text = throwing_with_lots_of_coverage
        corpus_size, total = len(shared), copy.deepcopy(shared.total_coverage)
        code = env._state.generate_program_code()

        for _ in range(10):
            cost = shared.execution_cost()
            reward = mutate_once(env)
            # It found nothing, so it just costs an execution
            assert reward == pytest.approx(-cost)
            assert env._state.generate_program_code() == code
            assert env._state.exec_data.error == JSError.NoError

        assert env.num_rollbacks == 10
        # Its coverage counts for nothing, so the program isn't added either
        assert len(shared) == corpus_size
        assert shared.total_coverage == total

    def test_target_is_in_the_restored_program(self):
        env, _, engine = make_env()
        env.reset(seed=0)
        engine.execute_text = throwing_with_lots_of_coverage
        mutate_once(env)

        node = env._state.target_node
        while node.parent is not None:
            assert any(child is node for child in node.parent.children())
            node = node.parent
        assert node is env._state.root

    def test_program_which_already_throws_is_still_mutated(self):
        env, shared, engine = make_env()
        env.reset(seed=0)
        env._state.exec_data.error = JSError.TypeError
        engine.execute_text = throwing_with_lots_of_coverage
        corpus_size = len(shared)

        cost = shared.execution_cost()
        reward = mutate_once(env)
        assert reward == pytest.approx(-cost)
        # The mutated program was kept: the state has its execution
        assert env.num_rollbacks == 0
        assert env._state.exec_data.coverage.hit_edges == NUM_EDGES
        # It increased its own coverage, but throwing programs never join the corpus
        assert len(shared) == corpus_size


class TestTimeouts:
    def test_timeout_is_not_added_or_rewarded(self):
        env, shared, engine = make_env()
        env.reset(seed=0)

        def timeout_with_lots_of_coverage(code, lib_path=None):
            edges = np.full(NUM_EDGES // 8, 0xFF, dtype=np.uint8)  # every edge "hit"
            return ExecutionData(Coverage(NUM_EDGES, edges), JSError.TimeoutError, "")

        engine.execute_text = timeout_with_lots_of_coverage
        corpus_size, total = len(shared), copy.deepcopy(shared.total_coverage)
        baseline = env._state.exec_data
        code = env._state.generate_program_code()

        for _ in range(50):
            valid = [a for a in (FuzzingAction.REPLACE, FuzzingAction.ADD) if env.action_mask()[a]]
            if not valid:
                env.step(FuzzingAction.MOVE_UP)
                continue
            _, reward, terminated, truncated, _ = env.step(valid[0])
            if terminated or truncated:
                break
            # Only the size penalty (0 for these small programs), never a coverage reward
            assert reward <= 0

        assert len(shared) == corpus_size
        assert shared.total_coverage == total
        # Every mutation was undone, so the episode is back on the last program
        # which completed
        assert env._state.exec_data.coverage == baseline.coverage
        assert env._state.generate_program_code() == code

    def test_prune_removes_timed_out_entries(self):
        env, shared, _ = make_env()
        timed_out = copy.deepcopy(shared.entries[0].state)
        timed_out.exec_data = ExecutionData(Coverage(), JSError.TimeoutError, "")
        with shared.lock:
            shared.add(timed_out)

        removed = shared.prune(
            lambda s: s.exec_data is not None and s.exec_data.error == JSError.TimeoutError
        )
        assert removed == 1
        assert all(e.state.exec_data.error != JSError.TimeoutError for e in shared.entries)


class TestSeedSelection:
    def _shared(self, times, selection="fast"):
        env, shared, _ = make_env()
        states = [shared.entries[i % len(shared.entries)].state for i in range(len(times))]
        fresh = SharedFuzzingState(states, shared.total_coverage, seed_selection=selection)
        for entry, t in zip(fresh.entries, times):
            entry.exec_time = t
        return fresh

    def test_faster_programs_are_picked_more(self):
        shared = self._shared([0.05, 0.1, 0.4])
        with shared.lock:
            weights = shared.selection_weights()
        # median 0.1: twice as fast -> 2x, four times slower -> clipped at 1/4
        assert weights.tolist() == [2.0, 1.0, 0.25]

        rng = np.random.default_rng(0)
        picks = [shared.entries.index(shared.sample(rng)) for _ in range(3000)]
        counts = np.bincount(picks, minlength=3)
        assert counts[0] > counts[1] > counts[2]

    def test_weights_are_clipped_and_unknown_gets_median(self):
        shared = self._shared([0.001, 0.1, None, 10.0])
        with shared.lock:
            weights = shared.selection_weights()
        assert weights[0] == 4.0 and weights[3] == 0.25
        assert weights[2] == 1.0

    def test_uniform_ignores_times(self):
        shared = self._shared([0.001, 10.0], selection="uniform")
        rng = np.random.default_rng(0)
        picks = [shared.entries.index(shared.sample(rng)) for _ in range(2000)]
        assert abs(np.mean(picks) - 0.5) < 0.05

    def test_exec_time_moving_average(self):
        shared = self._shared([None])
        entry = shared.entries[0]
        shared.episode_finished(entry, False, 0.2)
        assert entry.exec_time == 0.2
        shared.episode_finished(entry, True, 0.4)
        assert abs(entry.exec_time - (0.2 + 0.3 * 0.2)) < 1e-9

    def test_restores_checkpoints_without_timing(self):
        shared = self._shared([0.1, 0.2])
        snapshot = shared.snapshot()
        snapshot["entries"] = [(state, sel) for state, sel, _ in snapshot["entries"]]
        restored = SharedFuzzingState([], Coverage(), seed_selection="fast")
        restored.restore(snapshot)
        assert [e.exec_time for e in restored.entries] == [None, None]

    def test_env_reports_episode_exec_time(self):
        env, shared, _ = make_env()
        env.reset(seed=0)
        entry = env._entry
        for _ in range(20):
            valid = [a for a in (FuzzingAction.REPLACE, FuzzingAction.ADD) if env.action_mask()[a]]
            env.step(valid[0] if valid else FuzzingAction.MOVE_UP)
        executed = env._episode_execs
        env.reset()
        assert executed > 0 and entry.exec_time is not None and entry.exec_time >= 0
