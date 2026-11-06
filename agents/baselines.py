"""
baselines.py  --  Phase 6: two agents to compare Q-learning against.

RandomAgent   lower bound: picks a random action every step.
AStarAgent    upper bound: knows the whole map, plans the optimal path
              with A*, then follows it. Q-learning never sees the map, so
              matching A*'s path length means it learned the optimum.

Every agent exposes the same two methods so evaluation.py can treat them
alike:
    reset()      called at the start of each episode
    act(state)   returns an action (no exploration - this is the final policy)
"""

import heapq

import numpy as np

from env.grid_env import ACTIONS


class RandomAgent:
    def __init__(self, n_actions=len(ACTIONS), seed=0):
        self.n_actions = n_actions
        self.rng = np.random.default_rng(seed)

    def reset(self):
        pass

    def act(self, state):
        return int(self.rng.integers(self.n_actions))


# ----------------------------------------------------------------------
# A* search
# ----------------------------------------------------------------------
def manhattan(a, b):
    """Admissible heuristic for 4-connected grids: never overestimates."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def plan_path(env, heuristic=manhattan):
    """
    A* from env.start to env.goal using env.next_position for neighbours,
    so the planner obeys exactly the same movement rules as the robot.

    Returns (path, actions): path is a list of positions starting at
    env.start and ending at env.goal; actions[i] moves path[i] -> path[i+1].
    Returns (None, None) if the goal is unreachable.

    With heuristic=lambda a, b: 0 this is plain uniform-cost search
    (Dijkstra / BFS on a unit-cost grid): same answer, more cells expanded.
    """
    start, goal = env.start, env.goal
    frontier = [(heuristic(start, goal), 0, start)]   # (f, g, position)
    came_from = {start: None}                         # position -> (prev, action)
    best_g = {start: 0}

    while frontier:
        _, g, current = heapq.heappop(frontier)
        if current == goal:
            break
        if g > best_g[current]:                       # stale queue entry
            continue
        for action in ACTIONS:
            nxt, _ = env.next_position(current, action)
            if nxt == current:                        # wall or obstacle
                continue
            new_g = g + 1
            if nxt not in best_g or new_g < best_g[nxt]:
                best_g[nxt] = new_g
                came_from[nxt] = (current, action)
                heapq.heappush(frontier, (new_g + heuristic(nxt, goal), new_g, nxt))

    if goal not in came_from:
        return None, None

    # Walk the breadcrumbs backwards from the goal.
    path, actions = [goal], []
    node = goal
    while came_from[node] is not None:
        prev, action = came_from[node]
        path.append(prev)
        actions.append(action)
        node = prev
    return path[::-1], actions[::-1]


class AStarAgent:
    def __init__(self, env, heuristic=manhattan):
        self.path, self.actions = plan_path(env, heuristic)
        if self.actions is None:
            raise ValueError("A*: goal is unreachable on this map")
        self.i = 0

    def reset(self):
        self.i = 0

    def act(self, state):
        # Deterministic env, so just replay the plan. If something pushed
        # us off-plan, stop rather than wander (keeps the metric honest).
        if self.i >= len(self.actions):
            return self.actions[-1]
        action = self.actions[self.i]
        self.i += 1
        return action
