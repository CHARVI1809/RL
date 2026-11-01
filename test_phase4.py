"""
Phase 4 tests. Run from the project root:   python test_phase4.py
"""

import numpy as np

from env.grid_env import GridEnv, ACTIONS, REWARD_GOAL, REWARD_STEP
from agents.q_learning import QLearningAgent, train, greedy_rollout


def test_q_table_shape_and_init():
    agent = QLearningAgent(grid_size=10)
    assert agent.Q.shape == (10, 10, 4)
    assert np.all(agent.Q == 0)


def test_greedy_picks_best_action():
    agent = QLearningAgent(grid_size=10, epsilon=0.0)
    agent.Q[3, 3] = [1.0, 5.0, 2.0, -1.0]          # DOWN is best
    for _ in range(20):
        assert agent.choose_action((3, 3)) == 1


def test_greedy_breaks_ties_randomly():
    """All-zero Q-table: every action must come up, not just action 0."""
    agent = QLearningAgent(grid_size=10, epsilon=0.0)
    seen = {agent.choose_action((0, 0)) for _ in range(200)}
    assert seen == set(ACTIONS), f"only saw {seen}"


def test_full_exploration_is_uniform_ish():
    agent = QLearningAgent(grid_size=10, epsilon=1.0)
    agent.Q[0, 0] = [100.0, 0, 0, 0]                   # greedy would pick 0
    counts = np.bincount([agent.choose_action((0, 0)) for _ in range(4000)],
                         minlength=4)
    assert all(counts > 700), counts                  # expect ~1000 each


def test_update_math_non_terminal():
    agent = QLearningAgent(grid_size=10, alpha=0.5, gamma=0.9)
    agent.Q[1, 1] = [10.0, 0.0, 0.0, 0.0]             # max Q(next) = 10
    # target = -1 + 0.9 * 10 = 8 ; Q <- 0 + 0.5 * (8 - 0) = 4
    agent.update(state=(0, 0), action=3, reward=-1.0,
                 next_state=(1, 1), terminal=False)
    assert np.isclose(agent.Q[0, 0, 3], 4.0)
    assert np.all(agent.Q[0, 0, :3] == 0)             # other actions untouched


def test_update_math_terminal():
    agent = QLearningAgent(grid_size=10, alpha=0.5, gamma=0.9)
    agent.Q[9, 9] = [999.0] * 4                        # must be ignored
    agent.update(state=(9, 8), action=3, reward=REWARD_GOAL,
                 next_state=(9, 9), terminal=True)
    assert np.isclose(agent.Q[9, 8, 3], 0.5 * REWARD_GOAL)


def test_epsilon_decays_to_floor():
    agent = QLearningAgent(grid_size=10, epsilon=1.0,
                           epsilon_min=0.05, epsilon_decay=0.5)
    values = []
    for _ in range(10):
        agent.decay_epsilon()
        values.append(agent.epsilon)
    assert values[0] == 0.5 and values[1] == 0.25
    assert values[-1] == 0.05                          # floored, not below
    assert all(a >= b for a, b in zip(values, values[1:]))


def test_training_runs_and_learns():
    env = GridEnv()
    agent = QLearningAgent(grid_size=env.size, seed=42)
    hist = train(env, agent, n_episodes=600, log_every=0)

    # bookkeeping
    assert all(len(v) == 600 for v in hist.values())
    assert np.isfinite(agent.Q).all()
    assert np.any(agent.Q != 0)

    # the start cell should now look worse than cells next to the goal
    assert agent.Q[9, 8].max() > agent.Q[0, 0].max()

    # late training should be much better than early training
    early = np.mean(hist["reward"][:50])
    late = np.mean(hist["reward"][-50:])
    print(f"      avg reward: first 50 eps = {early:.1f}, last 50 eps = {late:.1f}")
    assert late > early + 50

    # the greedy policy must reach the goal, and on the shortest path (18)
    path, success = greedy_rollout(env, agent)
    print(f"      greedy path length = {len(path) - 1} steps, success = {success}")
    assert success
    assert len(path) - 1 == 18


if __name__ == "__main__":
    tests = [v for k, v in globals().items() if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS  {t.__name__}")
    print(f"\nAll {len(tests)} Phase 4 tests passed.")
