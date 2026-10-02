from __future__ import annotations

from collections import Counter
import copy
import itertools
import logging
from pathlib import Path
import pickle
import threading
import time
from typing import Any, Optional

import gymnasium as gym
from gymnasium import spaces
from js_ast.analysis import count_statements
from js_ast.analysis import scope_analysis
from js_ast.nodes import Node
import numpy as np
from rl.fuzzing_action import FuzzingAction
from rl.program_state import ProgramState
from transformer.tokenizer import ASTTokenizer

from utils.js_engine import Coverage
from utils.js_engine import Engine
from utils.js_engine import ExecutionData
from utils.js_engine import JSError


# Rewards. Only finding something new for the run counts: rewarding a program
# for increasing its own coverage let the agent farm reward by removing code and
# adding it back. Every execution costs the run's recent average reward per
# execution, so a mutation is worth making when it beats what the fuzzer
# currently achieves on average, and ending the episode (worth 0) is right once
# the program stops paying off. The cost tracks the actual rate, so it falls as
# new coverage gets harder to find and mutating never becomes pointless.
COVERAGE_REWARD = 2
CRASH_REWARD = 3
# For an action that can't be applied or a mutation which no longer generates
FAILED_ACTION_REWARD = -1
# Rate of new coverage per execution assumed before any are measured (about what
# random mutations of the seeds achieve), and how quickly the estimate moves
INITIAL_COVERAGE_RATE = 0.03
COVERAGE_RATE_SMOOTHING = 1e-4
PENALTY_STATEMENTS = 100
MAX_STATEMENTS = 1000
MAX_FRAGMENT_SEQ_LEN = 512  # Maximum length of the AST fragment sequence

MUTATIONS = (
    FuzzingAction.REPLACE,
    FuzzingAction.ADD,
    FuzzingAction.REMOVE,
    FuzzingAction.MODIFY,
)


class CorpusEntry:
    __slots__ = ["state", "selections_without_increase", "exec_time"]

    def __init__(self, state: ProgramState):
        self.state = state
        # Number of episodes started from this program since it last led to a
        # total coverage increase
        self.selections_without_increase = 0
        # Moving average of the mean execution time (seconds, timeouts counting
        # their full duration) of episodes started from this program. None until
        # the program has been used.
        self.exec_time: Optional[float] = None


SEED_SELECTIONS = ("uniform", "fast")


class SharedFuzzingState:
    """The corpus and total coverage of a fuzzing run. Shared by every environment
    (one per engine worker) so they all build on each other's progress.

    Programs to start episodes from are picked uniformly, or with
    seed_selection="fast" in proportion to how quickly their episodes execute (as
    AFL favours fast inputs): weight = median time / program's time, clipped to
    [1/4, 4]. Programs whose mutations are slow are picked less often, but never
    excluded, and programs not yet measured get the median weight.
    """

    EXEC_TIME_SMOOTHING = 0.3
    MIN_WEIGHT, MAX_WEIGHT = 0.25, 4.0

    def __init__(
        self,
        corpus: list[ProgramState],
        total_coverage: Coverage,
        max_eps_without_coverage_increase: int = 500,
        min_corpus_size: int = 16,
        seed_selection: str = "uniform",
    ):
        if seed_selection not in SEED_SELECTIONS:
            raise ValueError(f"Unknown seed selection {seed_selection}")
        self.entries = [CorpusEntry(state) for state in corpus]
        self.total_coverage = total_coverage
        self.max_eps_without_coverage_increase = max_eps_without_coverage_increase
        self.min_corpus_size = min_corpus_size
        self.seed_selection = seed_selection
        self.lock = threading.Lock()

        self.num_crashes = 0
        self.num_added = 0
        self.num_removed = 0
        # Moving average of how often an execution finds new coverage (or a crash)
        self.coverage_rate = INITIAL_COVERAGE_RATE

    def __len__(self):
        return len(self.entries)

    def snapshot(self) -> dict[str, Any]:
        """The corpus and coverage, for checkpointing. Stored program states are
        never modified once in the corpus, so a shallow copy of the list is enough
        and the lock is only held briefly."""
        with self.lock:
            return {
                "entries": [
                    (entry.state, entry.selections_without_increase, entry.exec_time)
                    for entry in self.entries
                ],
                "total_coverage": copy.deepcopy(self.total_coverage),
                "num_crashes": self.num_crashes,
                "num_added": self.num_added,
                "num_removed": self.num_removed,
                "coverage_rate": self.coverage_rate,
            }

    def prune(self, should_remove) -> int:
        """Removes corpus entries whose program state matches `should_remove`,
        always keeping at least min_corpus_size entries."""
        with self.lock:
            kept = [e for e in self.entries if not should_remove(e.state)]
            if len(kept) < self.min_corpus_size:
                return 0
            removed = len(self.entries) - len(kept)
            self.entries = kept
            self.num_removed += removed
            return removed

    def restore(self, snapshot: dict[str, Any]):
        with self.lock:
            self.entries = []
            for state, selections, *rest in snapshot["entries"]:
                entry = CorpusEntry(state)
                entry.selections_without_increase = selections
                # Checkpoints from before timing was recorded have no exec_time
                entry.exec_time = rest[0] if rest else None
                self.entries.append(entry)
            self.total_coverage = snapshot["total_coverage"]
            self.num_crashes = snapshot["num_crashes"]
            self.num_added = snapshot["num_added"]
            self.num_removed = snapshot["num_removed"]
            self.coverage_rate = snapshot.get("coverage_rate", INITIAL_COVERAGE_RATE)

    def selection_weights(self) -> np.ndarray:
        """Relative probability of picking each entry (caller must hold the lock)."""
        times = np.array(
            [np.nan if e.exec_time is None else e.exec_time for e in self.entries]
        )
        known = times[~np.isnan(times)]
        if len(known) == 0:
            return np.ones(len(times))
        median = float(np.median(known))
        times = np.where(np.isnan(times), median, times)
        return np.clip(
            median / np.maximum(times, 1e-6), self.MIN_WEIGHT, self.MAX_WEIGHT
        )

    def sample(self, rng: np.random.Generator) -> CorpusEntry:
        with self.lock:
            if self.seed_selection == "uniform":
                return self.entries[rng.integers(len(self.entries))]
            weights = self.selection_weights()
            return self.entries[rng.choice(len(self.entries), p=weights / weights.sum())]

    def execution_cost(self) -> float:
        """The average reward per execution, charged to every execution."""
        return COVERAGE_REWARD * self.coverage_rate

    def record_execution(self, found: bool):
        """Caller must hold the lock."""
        self.coverage_rate += COVERAGE_RATE_SMOOTHING * (float(found) - self.coverage_rate)

    def add(self, state: ProgramState):
        # Caller must hold the lock
        self.entries.append(CorpusEntry(state))
        self.num_added += 1

    def episode_finished(
        self,
        entry: CorpusEntry,
        total_coverage_increased: bool,
        mean_exec_time: Optional[float] = None,
    ):
        with self.lock:
            if mean_exec_time is not None:
                if entry.exec_time is None:
                    entry.exec_time = mean_exec_time
                else:
                    entry.exec_time += self.EXEC_TIME_SMOOTHING * (
                        mean_exec_time - entry.exec_time
                    )

            if total_coverage_increased:
                entry.selections_without_increase = 0
                return

            entry.selections_without_increase += 1
            if (
                entry.selections_without_increase
                >= self.max_eps_without_coverage_increase
                and len(self.entries) > self.min_corpus_size
            ):
                # Remove by identity as other environments may have changed the list
                for i, other in enumerate(self.entries):
                    if other is entry:
                        self.entries.pop(i)
                        self.num_removed += 1
                        logging.debug("Removed program from corpus")
                        break


class FuzzingEnv(gym.Env[tuple[list[int], list[int]], np.int64]):
    metadata = {}

    _save_ids = itertools.count()

    def __init__(
        self,
        shared: SharedFuzzingState,
        subtrees: dict[str, list[Node]],
        engine: Engine,
        tokenizer: ASTTokenizer,
        interesting_folder: Optional[Path] = None,
        max_mutations: int = 50,
        max_steps: int = 200,
        penalty_statements: int = PENALTY_STATEMENTS,
        max_statements: int = MAX_STATEMENTS,
        worker_id: int = 0,
        render_mode: Optional[str] = None,
    ):
        self.action_space = spaces.Discrete(len(FuzzingAction))
        self.observation_space = spaces.Tuple(
            (
                spaces.Box(
                    low=0,
                    high=len(tokenizer.vocab) - 1,
                    shape=(MAX_FRAGMENT_SEQ_LEN,),
                    dtype=np.int64,
                ),
                spaces.Box(
                    low=0,
                    high=len(tokenizer.vocab) - 1,
                    shape=(MAX_FRAGMENT_SEQ_LEN,),
                    dtype=np.int64,
                ),
            )
        )
        self.render_mode = render_mode

        self.shared = shared
        self.subtrees = subtrees
        self.engine = engine
        self.tokenizer = tokenizer
        self.interesting_folder = interesting_folder
        self.worker_id = worker_id

        self._state: ProgramState
        self._entry: Optional[CorpusEntry] = None
        # Execution time of the current episode, to rate its starting program
        self._episode_exec_time = 0.0
        self._episode_execs = 0

        self.num_mutations = 0  # number of mutations performed this episode
        self.num_steps = 0  # number of actions (including moves) this episode
        self.max_mutations = max_mutations  # max number of mutations to perform
        # Moves don't count as mutations, so also cap the total number of actions
        self.max_steps = max_steps
        # number of statements after which to apply penalty
        self.penalty_statements = penalty_statements
        self.max_statements = max_statements  # max number of statements in a program
        self._num_statements = 0

        self.total_coverage_increased = False  # whether the total coverage increased

        # Bumped whenever the AST changes, used to cache tokenised observations
        self._version = 0
        self._token_cache: dict[tuple[int, int], list[int]] = {}

        # Statistics
        self.total_executions = 0
        self.total_actions = 0
        self.num_failed_actions = 0
        self.num_rollbacks = 0  # mutations undone because they broke the program
        self.num_crashes = 0
        self.exec_errors: Counter[str] = Counter()
        self.action_counts: Counter[str] = Counter()
        self.exec_time = 0.0
        self.action_time = 0.0
        self.code_gen_time = 0.0
        self.reset_time = 0.0
        self.timeout_time = 0.0  # execution time spent on runs that timed out

    @property
    def total_coverage(self) -> Coverage:
        return self.shared.total_coverage

    _STATS = [
        "total_executions",
        "total_actions",
        "num_failed_actions",
        "num_rollbacks",
        "num_crashes",
        "exec_errors",
        "action_counts",
        "exec_time",
        "action_time",
        "code_gen_time",
        "reset_time",
        "timeout_time",
    ]

    def stats_state(self) -> dict[str, Any]:
        """Statistics counters, for checkpointing."""
        return {name: copy.copy(getattr(self, name)) for name in self._STATS}

    def add_stats(self, stats: dict[str, Any]):
        """Adds counters from a checkpoint (from one or more workers) to this one's."""
        for name in self._STATS:
            if name in stats:
                setattr(self, name, getattr(self, name) + stats[name])

    def save_current_state(self, save_type: str, exec_data: ExecutionData) -> None:
        if self.interesting_folder is None:
            return

        code = self._state.generate_program_code()
        if code is None:
            return

        name = f"{int(time.time())}_{self.worker_id}_{next(FuzzingEnv._save_ids)}_{save_type}"

        with open(self.interesting_folder / f"{name}.js", "w") as f:
            f.write(code)

        with open(self.interesting_folder / f"{name}.ps", "wb") as f:
            pickle.dump(self._state, f)

        with open(self.interesting_folder / f"{name}.txt", "w") as f:
            f.write(f"error: {exec_data.error.value}\n")
            f.write(f"lib: {self._state.lib_path}\n")
            f.write("--- stdout ---\n" + exec_data.out)
            f.write("\n--- stderr ---\n" + exec_data.err)

    def _tokenize(self, node: Node) -> list[int]:
        key = (id(node), self._version)
        tokens = self._token_cache.get(key)
        if tokens is None:
            if len(self._token_cache) > 256:
                self._token_cache.clear()
            tokens = self.tokenizer.tokenize(node)
            self._token_cache[key] = tokens
        return tokens

    def _get_obs(self) -> tuple[list[int], list[int]]:
        return (
            self._tokenize(self._state.get_target_node()),
            self._tokenize(self._state.get_context_node()),
        )

    def action_mask(self) -> np.ndarray:
        return np.array(self._state.valid_actions(self.subtrees), dtype=bool)

    def _get_info(self) -> dict[str, Any]:
        return {"action_mask": self.action_mask()}

    def _get_reward(self, exec_data: ExecutionData) -> float:
        penalty = min(0, 1 - self._num_statements / self.penalty_statements)

        save_type: Optional[str] = None
        with self.shared.lock:
            new_total_coverage = exec_data.coverage | self.shared.total_coverage

            if exec_data.is_crash():
                self.shared.total_coverage = new_total_coverage
                self.shared.num_crashes += 1
                self.num_crashes += 1
                save_type = "crash"
                reward = CRASH_REWARD + penalty
            elif (
                exec_data.error == JSError.NoError
                and new_total_coverage != self.shared.total_coverage
            ):
                # new test case increased its own coverage and the total coverage.
                # Only programs which run to the end count: one that throws stops
                # part-way, so it is a poor seed (mutating the code after the throw
                # does nothing) and a clean program reaching the same edges later
                # should still be rewarded.
                self.total_coverage_increased = True
                logging.info(
                    f"Coverage increased from {self.shared.total_coverage} to {new_total_coverage}"
                )
                # Store a copy, as this state will keep being mutated in this episode
                self.shared.add(copy.deepcopy(self._state))
                self.shared.total_coverage = new_total_coverage
                save_type = "coverage"
                reward = COVERAGE_REWARD + penalty
            else:
                reward = penalty

            reward -= self.shared.execution_cost()
            self.shared.record_execution(save_type is not None)

        if save_type == "crash":
            logging.info(f"Crash detected: {exec_data.err[-2000:]}")
        if save_type is not None:
            self.save_current_state(save_type, exec_data)

        return reward

    def _get_truncated(self) -> bool:
        return (
            self.num_mutations >= self.max_mutations
            or self.num_steps >= self.max_steps
            or self._num_statements >= self.max_statements
        )

    def _finish_step(
        self, reward: float, terminated: bool
    ) -> tuple[tuple[list[int], list[int]], float, bool, bool, dict[str, Any]]:
        truncated = not terminated and self._get_truncated()
        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[tuple[list[int], list[int]], dict[str, Any]]:
        # We need the following line to seed self.np_random
        super().reset(seed=seed, options=options)

        if self._entry is not None:
            self.shared.episode_finished(
                self._entry,
                self.total_coverage_increased,
                self._episode_exec_time / self._episode_execs if self._episode_execs else None,
            )

        reset_start = time.perf_counter()
        self._entry = self.shared.sample(self.np_random)
        self._state = copy.deepcopy(self._entry.state)
        self._version += 1

        scope_analysis(self._state.root)

        # Initialise state as random child of the root node
        self._state.move_down()
        self.num_mutations = 0
        self.num_steps = 0
        self._num_statements = count_statements(self._state.root)
        self._episode_exec_time = 0.0
        self._episode_execs = 0
        self.total_coverage_increased = False

        logging.debug("Starting new episode")
        self.reset_time += time.perf_counter() - reset_start

        return self._get_obs(), self._get_info()

    def step(
        self, action: np.int64 | int
    ) -> tuple[tuple[list[int], list[int]], float, bool, bool, dict[str, Any]]:
        action = FuzzingAction(int(action))
        logging.debug(f"Number of mutations: {self.num_mutations}, action: {action}")

        self.total_actions += 1
        self.num_steps += 1
        self.action_counts[str(action)] += 1

        start_time = time.perf_counter()
        # To undo a mutation which breaks the program. Copying is cheap next to
        # executing it (well under a millisecond for most programs).
        before = copy.deepcopy(self._state) if action in MUTATIONS else None
        match action:
            case FuzzingAction.MOVE_UP | FuzzingAction.MOVE_DOWN | FuzzingAction.MOVE_LEFT | FuzzingAction.MOVE_RIGHT:
                return self._move(action)
            case FuzzingAction.END:
                return self._finish_step(0, terminated=True)
            case FuzzingAction.REPLACE:
                new_node, changed = self._state.replace(self.subtrees)
            case FuzzingAction.ADD:
                new_node, changed = self._state.add(self.subtrees)
            case FuzzingAction.REMOVE:
                new_node, changed = self._state.remove(self.subtrees)
            case FuzzingAction.MODIFY:
                changed = self._state.modify()
                new_node = self._state.target_node
        self.action_time += time.perf_counter() - start_time

        if not changed:
            self.num_failed_actions += 1
            return self._finish_step(FAILED_ACTION_REWARD, terminated=False)

        self._version += 1
        self.num_mutations += 1
        self._state.target_node = new_node
        self._num_statements = count_statements(self._state.root)

        start = time.perf_counter()
        code = self._state.generate_program_code()
        self.code_gen_time += time.perf_counter() - start

        if not code:
            self._roll_back(before)
            return self._finish_step(FAILED_ACTION_REWARD, terminated=False)

        start = time.perf_counter()
        exec_data = self.engine.execute_text(code, self._state.lib_path)
        exec_duration = time.perf_counter() - start
        self.exec_time += exec_duration
        self._episode_exec_time += exec_duration
        self._episode_execs += 1
        self.total_executions += 1

        if not exec_data:
            return self._finish_step(-1, terminated=True)

        self.exec_errors[exec_data.error.value] += 1

        if exec_data.error == JSError.TimeoutError:
            self.timeout_time += exec_duration

        if exec_data.error == JSError.TimeoutError and not exec_data.is_crash():
            # A timed-out execution was killed part-way, so its coverage is partial
            # and says little about the program, and slow programs make poor seeds
            # (every mutation of them tends to time out too). As in AFL and Fuzzilli,
            # it is neither added to the corpus nor counted as coverage, and the
            # mutation is undone so the episode continues from the last program
            # which completed.
            self._roll_back(before)
            return self._finish_step(self._wasted_execution_reward(), terminated=False)

        if (
            exec_data.error != JSError.NoError
            and not exec_data.is_crash()
            and self._state_was_clean(before)
        ):
            # The mutation made a program that ran cleanly throw. Undo it: otherwise
            # every later mutation in the episode builds on a program that stops at
            # the exception, and mutations after it have no effect. Programs which
            # already threw (some seeds) are mutated as before, which may fix them.
            self._roll_back(before)
            return self._finish_step(self._wasted_execution_reward(), terminated=False)

        # Update before computing the reward so a copy added to the corpus carries the
        # coverage of the program it contains
        self._state.exec_data = exec_data
        reward = self._get_reward(exec_data)
        return self._finish_step(reward, terminated=exec_data.is_crash())

    def _wasted_execution_reward(self) -> float:
        """Reward for an execution that was undone (it timed out or broke the
        program): it found nothing, so it just costs an execution."""
        with self.shared.lock:
            cost = self.shared.execution_cost()
            self.shared.record_execution(False)
        return -cost

    @staticmethod
    def _state_was_clean(before: Optional[ProgramState]) -> bool:
        return before is not None and before.exec_data.error == JSError.NoError

    def _roll_back(self, before: Optional[ProgramState]):
        """Restores the program as it was before the last mutation."""
        if before is None:
            return
        self._state = before
        self._version += 1
        self._num_statements = count_statements(self._state.root)
        self.num_rollbacks += 1

    def _move(
        self, action: FuzzingAction
    ) -> tuple[tuple[list[int], list[int]], float, bool, bool, dict[str, Any]]:
        match action:
            case FuzzingAction.MOVE_UP:
                moved = self._state.move_up()
            case FuzzingAction.MOVE_DOWN:
                moved = self._state.move_down()
            case FuzzingAction.MOVE_LEFT:
                moved = self._state.move_left()
            case FuzzingAction.MOVE_RIGHT:
                moved = self._state.move_right()
            case _:
                raise ValueError(f"Invalid action: {action}")

        if not moved:
            self.num_failed_actions += 1

        return self._finish_step(0 if moved else -1, terminated=False)
