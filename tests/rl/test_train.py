import numpy as np
import torch

from rl.dqn import DQN
from rl.dqn import ReplayBuffer
from rl.fuzzing_action import FuzzingAction
from rl.train import double_dqn_targets
from rl.train import select_actions


N_ACTIONS = len(FuzzingAction)


class ConstantQ(torch.nn.Module):
    """A 'network' returning fixed Q-values, to make targets easy to check."""

    def __init__(self, values: list[float]):
        super().__init__()
        self.values = torch.tensor(values)

    def forward(self, x: torch.Tensor):
        return self.values.expand(x.shape[0], -1)


class TestDoubleDQNTargets:
    def test_no_bootstrap_after_episode_end(self):
        q = ConstantQ([5.0] * N_ACTIONS)
        rewards = torch.tensor([1.0, 1.0])
        dones = torch.tensor([False, True])
        masks = torch.ones(2, N_ACTIONS, dtype=torch.bool)

        targets = double_dqn_targets(q, q, rewards, torch.zeros(2, 4), dones, masks, 0.5)

        assert targets.tolist() == [1.0 + 0.5 * 5.0, 1.0]

    def test_argmax_only_over_valid_actions(self):
        # Action 0 has the highest value but is invalid in the next state
        policy = ConstantQ([100.0] + [float(i) for i in range(1, N_ACTIONS)])
        target = ConstantQ([-50.0] + [10.0 * i for i in range(1, N_ACTIONS)])
        masks = torch.ones(1, N_ACTIONS, dtype=torch.bool)
        masks[0, 0] = False

        targets = double_dqn_targets(
            policy,
            target,
            torch.tensor([0.0]),
            torch.zeros(1, 4),
            torch.tensor([False]),
            masks,
            1.0,
        )

        # policy's best valid action is the last one; its value comes from target
        assert targets.item() == 10.0 * (N_ACTIONS - 1)


class TestSelectActions:
    def test_only_valid_actions_are_selected(self):
        torch.manual_seed(0)
        policy = DQN(8, N_ACTIONS)
        states = torch.randn(64, 8)
        masks = torch.zeros(64, N_ACTIONS, dtype=torch.bool)
        valid = [FuzzingAction.REPLACE, FuzzingAction.MOVE_UP, FuzzingAction.END]
        masks[:, valid] = True

        for eps in (0.0, 0.5, 1.0):
            actions = select_actions(policy, states, masks, eps)
            assert set(actions.tolist()) <= set(int(a) for a in valid)

    def test_greedy_picks_best_valid_action(self):
        policy = ConstantQ([float(i) for i in range(N_ACTIONS)])
        masks = torch.ones(1, N_ACTIONS, dtype=torch.bool)
        masks[0, N_ACTIONS - 1] = False
        actions = select_actions(policy, torch.zeros(1, 4), masks, 0.0)
        assert actions.item() == N_ACTIONS - 2


class TestReplayBuffer:
    def test_ring_buffer_wraps(self):
        buffer = ReplayBuffer(4, 3, N_ACTIONS, torch.device("cpu"))
        for i in range(3):
            buffer.push(
                torch.full((2, 3), float(i)),
                torch.tensor([i, i]),
                torch.tensor([float(i), float(i)]),
                torch.full((2, 3), float(i)),
                torch.tensor([False, True]),
                torch.ones(2, N_ACTIONS, dtype=torch.bool),
                torch.full((2,), 0.9),
            )

        assert len(buffer) == 4
        # The third push wrapped around and overwrote the first one
        assert buffer.actions.tolist() == [2, 2, 1, 1]

        batch = buffer.sample(16)
        assert batch.states.dtype == torch.float32
        assert batch.states.shape == (16, 3)
        assert batch.next_masks.shape == (16, N_ACTIONS)


class TestNStep:
    def test_n_step_returns(self):
        from rl.train import NStepAccumulator

        acc = NStepAccumulator(3, 0.5)
        mask = np.ones(N_ACTIONS, dtype=bool)
        assert acc.add("s0", 0, 1.0, "s1", False, mask) == []
        assert acc.add("s1", 1, 2.0, "s2", False, mask) == []
        (t,) = acc.add("s2", 2, 4.0, "s3", False, mask)
        state, action, ret, next_state, done, _, discount = t
        # 1 + 0.5 * 2 + 0.25 * 4, bootstrapping from s3 with 0.5^3
        assert (state, action, ret, next_state, done, discount) == ("s0", 0, 3.0, "s3", False, 0.125)

    def test_episode_end_flushes_shorter_returns(self):
        from rl.train import NStepAccumulator

        acc = NStepAccumulator(3, 0.5)
        mask = np.ones(N_ACTIONS, dtype=bool)
        acc.add("s0", 0, 1.0, "s1", False, mask)
        ready = acc.add("s1", 1, 2.0, "end", True, mask)
        assert [(t[0], t[2], t[3], t[4], t[6]) for t in ready] == [
            ("s0", 1.0 + 0.5 * 2.0, "end", True, 0.0),
            ("s1", 2.0, "end", True, 0.0),
        ]
        # Nothing carries over into the next episode
        assert acc.add("t0", 0, 1.0, "t1", False, mask) == []

    def test_one_step_matches_plain_q_learning(self):
        from rl.train import NStepAccumulator

        acc = NStepAccumulator(1, 0.9)
        (t,) = acc.add("s0", 3, 2.0, "s1", False, np.ones(N_ACTIONS, dtype=bool))
        assert (t[2], t[3], t[6]) == (2.0, "s1", 0.9)


class TestPrioritisedReplay:
    def _buffer(self, alpha):
        buffer = ReplayBuffer(100, 2, N_ACTIONS, torch.device("cpu"), alpha=alpha)
        buffer.push(
            torch.zeros(100, 2),
            torch.arange(100) % N_ACTIONS,
            torch.zeros(100),
            torch.zeros(100, 2),
            torch.zeros(100, dtype=torch.bool),
            torch.ones(100, N_ACTIONS, dtype=torch.bool),
            torch.full((100,), 0.9),
        )
        return buffer

    def test_high_priority_transitions_are_sampled_more(self):
        torch.manual_seed(0)
        buffer = self._buffer(alpha=1.0)
        errors = torch.full((100,), 0.01)
        errors[7] = 10.0
        buffer.update_priorities(torch.arange(100), errors)

        batch = buffer.sample(1000, beta=1.0)
        assert (batch.indices == 7).float().mean() > 0.8
        # The over-sampled transition gets the smallest importance weight
        assert batch.weights[batch.indices == 7].max() < batch.weights[batch.indices != 7].min()
        assert batch.weights.max() <= 1.0

    def test_new_transitions_get_max_priority(self):
        buffer = self._buffer(alpha=1.0)
        buffer.update_priorities(torch.tensor([0]), torch.tensor([5.0]))
        buffer.push(
            torch.zeros(1, 2), torch.tensor([0]), torch.zeros(1), torch.zeros(1, 2),
            torch.zeros(1, dtype=torch.bool), torch.ones(1, N_ACTIONS, dtype=torch.bool),
            torch.tensor([0.9]),
        )
        assert buffer.priorities[0].item() == buffer.priorities.max().item() > 5.0

    def test_uniform_without_alpha(self):
        batch = self._buffer(alpha=0.0).sample(64)
        assert torch.all(batch.weights == 1)


class TestDueling:
    def test_q_values_combine_value_and_advantage(self):
        torch.manual_seed(0)
        net = DQN(8, N_ACTIONS, dueling=True)
        x = torch.randn(4, 8)
        q = net(x)
        assert q.shape == (4, N_ACTIONS)
        # Mean-centred advantages: the mean Q-value equals the state value
        h = torch.nn.functional.leaky_relu
        z = h(net.layer3(h(net.layer2(h(net.layer1(x))))))
        value = net.value5(h(net.value4(z))).squeeze(1)
        assert torch.allclose(q.mean(dim=1), value, atol=1e-5)
