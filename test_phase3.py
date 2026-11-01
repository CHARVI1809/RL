"""
Phase 3 tests. Run from the project root:   python test_phase3.py
"""

from collections import deque

from env.grid_env import (GridEnv, UP, DOWN, LEFT, RIGHT, ACTIONS,
                          REWARD_GOAL, REWARD_STEP, REWARD_COLLISION)


def test_ordinary_move_reward():
    env = GridEnv(); env.reset()
    pos, r, done, info = env.step(RIGHT)
    assert pos == (0, 1) and r == REWARD_STEP and not done


def test_wall_bump_reward():
    env = GridEnv(); env.reset()
    pos, r, done, info = env.step(UP)             # off the top edge
    assert pos == (0, 0) and r == REWARD_COLLISION and not done
    assert info["bumped_wall"]


def test_obstacle_bump_reward():
    env = GridEnv(); env.reset()
    env.robot_pos = (0, 1)
    pos, r, done, info = env.step(DOWN)           # into (1, 1)
    assert pos == (0, 1) and r == REWARD_COLLISION and not done
    assert info["bumped_obstacle"]


def test_goal_reward_and_done():
    env = GridEnv(); env.reset()
    env.robot_pos = (9, 8)
    pos, r, done, info = env.step(RIGHT)
    assert pos == env.goal and r == REWARD_GOAL and done


def test_reset_clears_step_count():
    env = GridEnv(); env.reset()
    for _ in range(5):
        env.step(RIGHT)
    assert env.step_count == 5
    env.reset()
    assert env.step_count == 0 and env.robot_pos == env.start


def test_step_limit_ends_episode():
    env = GridEnv(max_steps=7); env.reset()
    done = False
    for i in range(7):
        assert not done, f"episode ended early at step {i}"
        _, _, done, _ = env.step(UP)              # bump the wall forever
    assert done and env.step_count == 7


def test_scripted_route_total_reward():
    """9x RIGHT + 9x DOWN: 17 ordinary moves then the goal."""
    env = GridEnv(); env.reset()
    total, done = 0.0, False
    for a in [RIGHT] * 9 + [DOWN] * 9:
        assert not done
        _, r, done, _ = env.step(a)
        total += r
    assert done
    assert total == 17 * REWARD_STEP + REWARD_GOAL     # -17 + 100 = 83


def test_reward_values_are_consistent():
    """The numbers must actually encode what we want the agent to learn."""
    # ordering: collision worse than a step, step worse than nothing, goal good
    assert REWARD_COLLISION < REWARD_STEP < 0 < REWARD_GOAL

    # The goal must be worth more than the cost of walking to it, otherwise
    # the agent would rather stand still. Compute shortest path with BFS.
    env = GridEnv()
    dist = {env.start: 0}
    q = deque([env.start])
    while q:
        pos = q.popleft()
        for a in ACTIONS:
            nxt, _ = env.next_position(pos, a)
            if nxt not in dist:
                dist[nxt] = dist[pos] + 1
                q.append(nxt)
    shortest = dist[env.goal]
    print(f"      shortest path = {shortest} steps")
    assert REWARD_GOAL + (shortest - 1) * REWARD_STEP > 0


if __name__ == "__main__":
    tests = [v for k, v in globals().items() if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS  {t.__name__}")
    print(f"\nAll {len(tests)} Phase 3 tests passed.")
