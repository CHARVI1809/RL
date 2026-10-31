"""
Phase 1 tests. Run from the project root:   python test_phase1.py
No pytest needed - plain asserts.
"""

import os
from collections import deque

import matplotlib
matplotlib.use("Agg")   # draw to files, no window needed

from env.grid_env import GridEnv, FREE, OBSTACLE, DEFAULT_OBSTACLES


def test_grid_shape():
    env = GridEnv()
    assert env.grid.shape == (10, 10)
    assert env.grid.dtype.kind == "i"          # integer array


def test_obstacle_count():
    env = GridEnv()
    assert int(env.grid.sum()) == len(DEFAULT_OBSTACLES)
    for cell in DEFAULT_OBSTACLES:
        assert env.grid[cell] == OBSTACLE


def test_start_and_goal_are_free():
    env = GridEnv()
    assert env.grid[env.start] == FREE
    assert env.grid[env.goal] == FREE
    assert env.start != env.goal


def test_bounds_checks():
    env = GridEnv()
    assert env.in_bounds((0, 0))
    assert env.in_bounds((9, 9))
    assert not env.in_bounds((10, 0))
    assert not env.in_bounds((0, 10))
    assert not env.in_bounds((-1, 5))
    assert env.is_obstacle((1, 1))
    assert not env.is_obstacle((0, 0))
    assert not env.is_obstacle((-1, -1))      # outside -> not an obstacle
    assert env.is_free((0, 1))


def test_reset_puts_robot_on_start():
    env = GridEnv()
    env.robot_pos = (5, 5)                     # pretend it moved
    assert env.reset() == env.start
    assert env.robot_pos == env.start


def test_invalid_configs_are_rejected():
    for bad_kwargs in (
        {"start": (1, 1)},          # start on an obstacle
        {"goal": (10, 10)},         # goal outside grid
        {"obstacles": [(12, 0)]},   # obstacle outside grid
    ):
        try:
            GridEnv(**bad_kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"GridEnv accepted bad config {bad_kwargs}")


def test_a_path_exists():
    """BFS over free cells: the default map must be solvable."""
    env = GridEnv()
    queue, seen = deque([env.start]), {env.start}
    while queue:
        r, c = queue.popleft()
        if (r, c) == env.goal:
            return
        for nxt in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if env.is_free(nxt) and nxt not in seen:
                seen.add(nxt)
                queue.append(nxt)
    raise AssertionError("No path from start to goal - fix DEFAULT_OBSTACLES")


def test_render_saves_image():
    os.makedirs("results", exist_ok=True)
    path = "results/phase1_grid.png"
    if os.path.exists(path):
        os.remove(path)
    GridEnv().render(save_path=path, show=False)
    assert os.path.exists(path) and os.path.getsize(path) > 0


if __name__ == "__main__":
    tests = [v for k, v in globals().items() if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS  {t.__name__}")
    print(f"\nAll {len(tests)} Phase 1 tests passed.")
