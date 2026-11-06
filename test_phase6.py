"""
Phase 6 tests. Run from the project root:   python test_phase6.py
"""

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")

from env.grid_env import GridEnv, REWARD_GOAL, REWARD_STEP, DEFAULT_MAX_STEPS
from agents.q_learning import QLearningAgent, train
from agents.baselines import RandomAgent, AStarAgent, plan_path, manhattan
from evaluation import (run_episode, evaluate, compare,
                        convergence_episode, format_table)
from utils.plotting import plot_comparison
from utils.animate import animate_agents


def test_astar_finds_shortest_path():
    env = GridEnv()
    path, actions = plan_path(env)
    assert path[0] == env.start and path[-1] == env.goal
    assert len(actions) == len(path) - 1 == 18
    # replaying the actions through the real env must reproduce the path
    env.reset()
    for a, expected in zip(actions, path[1:]):
        pos, _, _, info = env.step(a)
        assert pos == expected and not (info["bumped_wall"] or info["bumped_obstacle"])


def test_astar_and_uniform_cost_agree():
    env = GridEnv()
    p_astar, _ = plan_path(env, heuristic=manhattan)
    p_ucs, _ = plan_path(env, heuristic=lambda a, b: 0)
    assert len(p_astar) == len(p_ucs)            # same optimal length


def test_astar_unreachable_goal():
    walled = [(0, 1), (1, 0)]                    # boxes the start in
    env = GridEnv(obstacles=walled)
    assert plan_path(env) == (None, None)
    try:
        AStarAgent(env)
    except ValueError:
        pass
    else:
        raise AssertionError("AStarAgent accepted an unsolvable map")


def test_run_episode_astar_metrics():
    env = GridEnv()
    m = run_episode(env, AStarAgent(env))
    assert m == {"success": True, "steps": 18,
                 "reward": 17 * REWARD_STEP + REWARD_GOAL, "collisions": 0}


def test_evaluate_random_agent_is_sane():
    env = GridEnv()
    m = evaluate(env, RandomAgent(seed=0), n_episodes=30)
    assert 0.0 <= m["success_rate"] <= 1.0
    assert 18 < m["avg_steps"] <= DEFAULT_MAX_STEPS
    assert m["avg_collisions"] > 0                # random walking bumps things
    print(f"      random agent: success {m['success_rate']:.0%}, "
          f"avg steps {m['avg_steps']:.0f}, avg collisions {m['avg_collisions']:.0f}")


def test_trained_agent_matches_astar():
    env = GridEnv()
    agent = QLearningAgent(grid_size=env.size, seed=5)
    hist = train(env, agent, n_episodes=800, log_every=0)
    q = evaluate(env, agent, n_episodes=20)
    a = evaluate(env, AStarAgent(env), n_episodes=20)
    assert q["success_rate"] == 1.0
    assert q["avg_steps"] == a["avg_steps"] == 18
    assert q["avg_collisions"] == 0
    assert np.isclose(q["avg_reward"], a["avg_reward"])
    # training-time collisions must drop
    assert len(hist["collisions"]) == 800
    assert np.mean(hist["collisions"][-100:]) < np.mean(hist["collisions"][:100])


def test_convergence_episode_math():
    # smoothed reward jumps from 0 to 100 at episode 11 and stays there
    hist = {"reward": [0.0] * 10 + [100.0] * 40}
    conv = convergence_episode(hist, window=1, tol=5)
    assert conv == 11
    flat = {"reward": [50.0] * 20}
    assert convergence_episode(flat, window=5) == 1


def test_compare_table_and_plot():
    env = GridEnv()
    results = compare(env, {"A*": AStarAgent(env), "Random": RandomAgent()}, n_episodes=5)
    table = format_table(results)
    assert table.count("\n") == 3 and "| A* |" in table and "100%" in table
    path = "results/_test_cmp.png"
    plot_comparison(results, save_path=path, show=False)
    assert os.path.exists(path) and os.path.getsize(path) > 0
    os.remove(path)


def test_run_episode_records_path_and_side_by_side_gif():
    env = GridEnv()
    a = run_episode(env, AStarAgent(env), record_path=True)
    r = run_episode(env, RandomAgent(seed=1), record_path=True)
    assert a["path"][0] == env.start and a["path"][-1] == env.goal
    assert len(a["path"]) - 1 == a["steps"] and len(r["path"]) - 1 == r["steps"]
    gif = "results/_test_side.gif"
    anim = animate_agents(env, {"A*": a["path"], "Random": r["path"]},
                          save_path=gif, interval_ms=50, max_frames=12)
    assert len(list(anim.new_frame_seq())) == 12
    assert os.path.exists(gif) and os.path.getsize(gif) > 0
    assert env.robot_pos == env.start
    os.remove(gif)


if __name__ == "__main__":
    tests = [v for k, v in globals().items() if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS  {t.__name__}")
    print(f"\nAll {len(tests)} Phase 6 tests passed.")
