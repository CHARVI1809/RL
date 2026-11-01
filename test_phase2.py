"""
Phase 2 tests. Run from the project root:   python test_phase2.py
"""

from env.grid_env import (GridEnv, UP, DOWN, LEFT, RIGHT, ACTIONS,
                          ACTION_DELTAS, OBSTACLE)


def test_each_action_moves_one_cell():
    env = GridEnv()
    env.robot_pos = (4, 5)                     # all four neighbours are free
    expected = {UP: (3, 5), DOWN: (5, 5), LEFT: (4, 4), RIGHT: (4, 6)}
    for action, target in expected.items():
        env.robot_pos = (4, 5)
        new_pos, _, _, info = env.step(action)
        assert new_pos == target, f"{action}: got {new_pos}, want {target}"
        assert not info["bumped_wall"] and not info["bumped_obstacle"]


def test_cannot_leave_grid():
    env = GridEnv()
    cases = [((0, 4), UP), ((9, 4), DOWN), ((4, 0), LEFT), ((4, 9), RIGHT)]
    for pos, action in cases:
        env.robot_pos = pos
        new_pos, _, _, info = env.step(action)
        assert new_pos == pos, f"{pos} + {action} left the grid: {new_pos}"
        assert info["bumped_wall"] and not info["bumped_obstacle"]


def test_corners():
    env = GridEnv()
    env.robot_pos = (0, 0)
    assert env.step(UP)[0] == (0, 0)
    assert env.step(LEFT)[0] == (0, 0)
    env.robot_pos = (9, 9)
    assert env.step(DOWN)[0] == (9, 9)
    assert env.step(RIGHT)[0] == (9, 9)


def test_obstacle_blocks_movement():
    env = GridEnv()
    env.robot_pos = (0, 1)                     # (1, 1) below is an obstacle
    new_pos, _, _, info = env.step(DOWN)
    assert new_pos == (0, 1)
    assert info["bumped_obstacle"] and not info["bumped_wall"]
    env.robot_pos = (2, 6)                     # (2, 7) to the right is a wall
    assert env.step(RIGHT)[0] == (2, 6)


def test_goal_detection():
    env = GridEnv()
    env.robot_pos = (9, 8)
    new_pos, _, _, info = env.step(RIGHT)
    assert new_pos == env.goal and info["reached_goal"]
    env.robot_pos = (0, 0)
    assert not env.step(RIGHT)[3]["reached_goal"]


def test_invalid_action_rejected():
    env = GridEnv()
    for bad in (4, -1, "up"):
        try:
            env.step(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"step() accepted invalid action {bad!r}")


def test_next_position_does_not_mutate_env():
    env = GridEnv()
    env.reset()
    env.next_position((0, 0), RIGHT)
    assert env.robot_pos == (0, 0)


def test_all_cells_all_actions():
    """Exhaustive sweep: from every free cell, every action must land on a
    free in-bounds cell that is either the same cell or an adjacent one."""
    env = GridEnv()
    checked = 0
    for r in range(env.size):
        for c in range(env.size):
            if env.grid[r, c] == OBSTACLE:
                continue
            for a in ACTIONS:
                new_pos, info = env.next_position((r, c), a)
                assert env.is_free(new_pos)
                dr, dc = ACTION_DELTAS[a]
                moved = new_pos == (r + dr, c + dc)
                stayed = new_pos == (r, c)
                assert moved or stayed
                # If we stayed, exactly one reason must be given.
                assert moved != (info["bumped_wall"] or info["bumped_obstacle"])
                checked += 1
    assert checked == (env.size ** 2 - int(env.grid.sum())) * len(ACTIONS)
    print(f"      swept {checked} (cell, action) pairs")


def test_scripted_route_reaches_goal():
    """Row 0 and col 9 are obstacle-free on the default map, so
    9x RIGHT then 9x DOWN must reach the goal without any bumps."""
    env = GridEnv()
    env.reset()
    route = [RIGHT] * 9 + [DOWN] * 9
    for i, a in enumerate(route):
        pos, _, _, info = env.step(a)
        assert not info["bumped_wall"] and not info["bumped_obstacle"], \
            f"bump at step {i}, pos {pos}"
    assert pos == env.goal and info["reached_goal"]


if __name__ == "__main__":
    tests = [v for k, v in globals().items() if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS  {t.__name__}")
    print(f"\nAll {len(tests)} Phase 2 tests passed.")
