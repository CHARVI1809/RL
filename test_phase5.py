"""
Phase 5 tests. Run from the project root:   python test_phase5.py
"""

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from env.grid_env import GridEnv, RIGHT, DOWN
from agents.q_learning import QLearningAgent, train, greedy_rollout
from utils.plotting import moving_average, plot_training
from utils.animate import animate_path


def test_moving_average_math():
    x = [1, 2, 3, 4, 5]
    # window 3: [1, (1+2)/2, (1+2+3)/3, (2+3+4)/3, (3+4+5)/3]
    assert np.allclose(moving_average(x, 3), [1, 1.5, 2, 3, 4])
    assert len(moving_average(x, 50)) == 5                # never longer
    assert np.allclose(moving_average(x, 1), x)           # window 1 = raw


def test_plot_training_writes_file():
    hist = {"reward": [-5, -3, 0, 2, 3], "steps": [50, 40, 30, 20, 18],
            "success": [0, 0, 1, 1, 1], "epsilon": [1, .9, .8, .7, .6]}
    os.makedirs("results", exist_ok=True)
    path = "results/_test_curves.png"
    plot_training(hist, window=2, save_path=path, show=False)
    assert os.path.exists(path) and os.path.getsize(path) > 0
    os.remove(path)


def test_render_with_path_and_policy():
    env = GridEnv()
    scripted = [(0, 0)] + [(0, c) for c in range(1, 10)] + [(r, 9) for r in range(1, 10)]
    policy = np.full((10, 10), RIGHT)
    path = "results/_test_path.png"
    env.render(path=scripted, policy=policy, save_path=path, show=False)
    assert os.path.exists(path) and os.path.getsize(path) > 0
    os.remove(path)


def test_render_into_existing_axes_does_not_close_it():
    env = GridEnv()
    fig, axes = plt.subplots(1, 2)
    env.render(ax=axes[0], show=False)
    env.render(ax=axes[1], show=False, path=[(0, 0), (0, 1)])
    assert plt.fignum_exists(fig.number)                  # still open
    plt.close(fig)


def test_learned_path_is_valid():
    """The greedy route must be physically possible: starts at S, ends at G,
    every step is to an adjacent free cell, and no cell is visited twice."""
    env = GridEnv()
    agent = QLearningAgent(grid_size=env.size, seed=1)
    train(env, agent, n_episodes=600, log_every=0)
    path, success = greedy_rollout(env, agent)

    assert success and path[0] == env.start and path[-1] == env.goal
    for a, b in zip(path, path[1:]):
        assert env.is_free(b)
        assert abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1   # one cell apart
    assert len(set(path)) == len(path)                    # no loops
    print(f"      route has {len(path) - 1} steps")


def test_animation_writes_gif_and_resets_env():
    env = GridEnv()
    scripted = [(0, 0), (0, 1), (0, 2)]
    path = "results/_test_anim.gif"
    animate_path(env, scripted, save_path=path, interval_ms=50)
    assert os.path.exists(path) and os.path.getsize(path) > 0
    assert env.robot_pos == env.start                     # tidied up
    os.remove(path)


def test_recorded_episode_paths():
    env = GridEnv()
    agent = QLearningAgent(grid_size=env.size, seed=3)
    hist = train(env, agent, n_episodes=30, log_every=0, record_episodes=(1, 7))
    assert set(hist["paths"]) == {1, 7}
    for ep, p in hist["paths"].items():
        assert p[0] == env.start
        assert len(p) - 1 == hist["steps"][ep - 1]       # one position per step
        for a, b in zip(p, p[1:]):                        # stay-put or adjacent
            assert abs(a[0] - b[0]) + abs(a[1] - b[1]) <= 1
            assert env.is_free(b)
        assert (p[-1] == env.goal) == hist["success"][ep - 1]


def test_animation_max_frames_truncates():
    env = GridEnv()
    long_path = [(0, 0)] * 50
    anim = animate_path(env, long_path, save_path=None, max_frames=10)
    assert len(list(anim.new_frame_seq())) == 10        # version-independent


if __name__ == "__main__":
    tests = [v for k, v in globals().items() if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS  {t.__name__}")
    print(f"\nAll {len(tests)} Phase 5 tests passed.")
