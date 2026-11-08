"""
Phase 7 tests. Run from the project root:   python test_phase7.py
"""

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")

from env.grid_env import UP, RIGHT, REWARD_COLLISION, REWARD_STEP
from env.dynamic_grid_env import DynamicGridEnv, MovingObstacle, DEFAULT_MOVERS
from agents.q_learning import QLearningAgent, train
from agents.baselines import AStarAgent
from evaluation import evaluate


def test_moving_obstacle_cycles():
    m = MovingObstacle([(2, 4), (2, 5), (2, 6), (2, 5)])
    assert m.period == 4
    assert [m.position(t) for t in range(6)] == [(2, 4), (2, 5), (2, 6), (2, 5), (2, 4), (2, 5)]
    try:
        MovingObstacle([(0, 0), (0, 2)])            # not adjacent
    except ValueError:
        pass
    else:
        raise AssertionError("non-adjacent patrol path accepted")


def test_dynamic_env_state_and_validation():
    env = DynamicGridEnv()
    assert env.state_shape == (10, 10, 4) and env.period == 4
    assert env.reset() == (0, 0, 0)
    assert env.mover_cells(0) == {(2, 4), (4, 9)}
    assert env.mover_cells(2) == {(2, 6), (6, 9)}
    for bad in ([MovingObstacle([(1, 1), (1, 2)])],      # on a wall
                [MovingObstacle([(9, 9), (9, 8)])]):     # covers the goal
        try:
            DynamicGridEnv(movers=bad)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid mover path accepted")
    # static walls still behave as before
    assert env.is_obstacle((1, 1)) and env.is_free((2, 4))


def _tiny():
    """Empty grid, one mover bouncing between (0,2) and (0,3)."""
    return DynamicGridEnv(obstacles=[], movers=[MovingObstacle([(0, 2), (0, 3)])])


def test_safe_pass_and_walk_into_mover():
    env = _tiny(); env.reset()                           # t=0: mover (0,2) -> (0,3)
    s, r, _, info = env.step(RIGHT)                      # (0,0) -> (0,1), safe
    assert s == (0, 1, 1) and r == REWARD_STEP and not info["bumped_mover"]
    s, r, _, info = env.step(RIGHT)                      # t=1: mover (0,3) -> (0,2): blocked
    assert s == (0, 1, 0) and r == REWARD_COLLISION and info["bumped_mover"]
    assert env.t == 2                                    # clock still advances


def test_swap_is_a_collision():
    env = _tiny(); env.reset()
    env.robot_pos, env.t = (0, 2), 1                     # mover at (0,3), moving to (0,2)
    s, r, _, info = env.step(RIGHT)                      # we try (0,2)->(0,3): swap
    assert s[:2] == (0, 2) and r == REWARD_COLLISION and info["bumped_mover"]


def test_mover_runs_into_waiting_robot():
    env = _tiny(); env.reset()
    env.robot_pos, env.t = (0, 2), 1                     # mover about to enter (0,2)
    s, r, _, info = env.step(UP)                         # UP hits the wall = waiting
    assert s[:2] == (0, 2) and r == REWARD_COLLISION and info["bumped_mover"]


def test_waiting_is_safe_when_no_mover_arrives():
    env = _tiny(); env.reset()
    env.robot_pos, env.t = (0, 1), 0                     # mover (0,2) -> (0,3), away from us
    s, r, _, info = env.step(UP)                         # wall bump only
    assert s[:2] == (0, 1) and r == REWARD_COLLISION
    assert info["bumped_wall"] and not info["bumped_mover"]


def test_render_with_movers():
    env = DynamicGridEnv(); env.reset()
    path = "results/_test_dyn.png"
    env.render(save_path=path, show=False)
    assert os.path.exists(path) and os.path.getsize(path) > 0
    os.remove(path)


def test_agent_with_phase_state():
    env = DynamicGridEnv()
    agent = QLearningAgent(state_shape=env.state_shape)
    assert agent.Q.shape == (10, 10, 4, 4)
    agent.update((0, 0, 0), RIGHT, -1.0, (0, 1, 1), terminal=False)
    assert agent.Q[0, 0, 0, RIGHT] == -0.1               # alpha * (-1 - 0)
    assert np.count_nonzero(agent.Q) == 1


def test_learns_to_avoid_patrols_and_static_astar_does_not():
    env = DynamicGridEnv()
    agent = QLearningAgent(state_shape=env.state_shape, seed=7, epsilon_decay=0.998)
    train(env, agent, n_episodes=2000, log_every=0)
    q = evaluate(env, agent, n_episodes=20)
    a = evaluate(env, AStarAgent(env), n_episodes=5)     # deterministic
    print(f"      Q-learning: steps {q['avg_steps']:.0f}, collisions {q['avg_collisions']:.1f} | "
          f"static A*: steps {a['avg_steps']:.0f}, collisions {a['avg_collisions']:.1f}")
    assert q["success_rate"] == 1.0 and q["avg_collisions"] == 0.0
    assert q["avg_steps"] <= 20                           # at most a short wait
    assert a["avg_collisions"] >= 1                      # map-only planner gets hit


if __name__ == "__main__":
    tests = [v for k, v in globals().items() if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS  {t.__name__}")
    print(f"\nAll {len(tests)} Phase 7 tests passed.")
