"""
grid_env.py  --  Phase 1: the grid world.

Conventions used throughout the project
---------------------------------------
* A position is a tuple (row, col).
  Row 0 is the TOP of the grid, col 0 is the LEFT.
* The grid is a 2-D NumPy array of integers:
      0 = FREE cell
      1 = OBSTACLE cell
* The robot itself is NOT stored in the grid array. Its position is a
  separate attribute (self.robot_pos), so the map never has to be
  "cleaned up" when the robot moves.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

FREE = 0
OBSTACLE = 1

# A fixed obstacle layout for the 10x10 grid.
# Chosen so that at least one path from (0, 0) to (9, 9) exists.
DEFAULT_OBSTACLES = [
    (1, 1), (1, 2), (1, 3),          # small wall near the start
    (3, 3), (4, 3), (5, 3), (6, 3),  # vertical wall in the middle-left
    (2, 7), (3, 7), (4, 7),          # vertical wall on the right
    (6, 5), (6, 6),                  # short horizontal wall
    (8, 1), (8, 2),                  # bottom-left blocker
    (8, 6), (8, 7),                  # blocker near the goal
]

# ----------------------------------------------------------------------
# Actions (Phase 2)
# ----------------------------------------------------------------------
# Actions are plain integers 0..3 so they can index a Q-table later.
UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3
ACTIONS = [UP, DOWN, LEFT, RIGHT]
ACTION_NAMES = {UP: "UP", DOWN: "DOWN", LEFT: "LEFT", RIGHT: "RIGHT"}

# How each action changes (row, col). Remember: row grows DOWNWARD.
ACTION_DELTAS = {
    UP:    (-1, 0),
    DOWN:  (+1, 0),
    LEFT:  (0, -1),
    RIGHT: (0, +1),
}

# ----------------------------------------------------------------------
# Rewards (Phase 3)
# ----------------------------------------------------------------------
# The three outcomes of a step are mutually exclusive: a bump means the
# robot did not move, so it cannot have reached the goal on that step.
REWARD_GOAL = 100.0        # reaching the goal cell
REWARD_STEP = -1.0         # any ordinary move (makes shorter paths better)
REWARD_COLLISION = -10.0   # trying to leave the grid or enter an obstacle

# Safety net so an episode cannot run forever during training.
DEFAULT_MAX_STEPS = 200


class GridEnv:
    """A static grid world with a start cell, a goal cell and obstacles."""

    def __init__(self, size=10, start=(0, 0), goal=(9, 9), obstacles=None,
                 max_steps=DEFAULT_MAX_STEPS):
        self.size = size
        self.max_steps = max_steps
        self.start = tuple(start)
        self.goal = tuple(goal)

        # Build the map: everything starts FREE, then we mark obstacles.
        self.grid = np.zeros((size, size), dtype=int)
        obstacles = DEFAULT_OBSTACLES if obstacles is None else obstacles
        for cell in obstacles:
            cell = tuple(cell)
            if not self.in_bounds(cell):
                raise ValueError(f"Obstacle {cell} is outside the grid")
            self.grid[cell] = OBSTACLE

        # Sanity checks: start and goal must be inside the grid and free.
        for name, cell in (("start", self.start), ("goal", self.goal)):
            if not self.in_bounds(cell):
                raise ValueError(f"{name} {cell} is outside the grid")
            if self.grid[cell] == OBSTACLE:
                raise ValueError(f"{name} {cell} is placed on an obstacle")

        self.robot_pos = self.start
        self.step_count = 0

    # ------------------------------------------------------------------
    # Small helper queries (will be used heavily in Phase 2)
    # ------------------------------------------------------------------
    def in_bounds(self, pos):
        """True if pos lies inside the grid."""
        r, c = pos
        return 0 <= r < self.size and 0 <= c < self.size

    def is_obstacle(self, pos):
        """True if pos is inside the grid AND contains an obstacle."""
        return self.in_bounds(pos) and self.grid[pos] == OBSTACLE

    def is_free(self, pos):
        """True if pos is inside the grid AND is not an obstacle."""
        return self.in_bounds(pos) and self.grid[pos] == FREE

    def reset(self):
        """Start a new episode: robot on the start cell, step counter at 0."""
        self.robot_pos = self.start
        self.step_count = 0
        return self.robot_pos

    # ------------------------------------------------------------------
    # Movement (Phase 2)
    # ------------------------------------------------------------------
    def next_position(self, pos, action):
        """
        Where would the robot end up if it took `action` from `pos`?

        Returns (new_pos, info) where info tells you WHY the move did or
        did not happen. This is a pure function: it does not change the
        environment, which makes it easy to test and to reuse in A* later.
        """
        if action not in ACTION_DELTAS:
            raise ValueError(f"Unknown action {action}; use one of {ACTIONS}")

        dr, dc = ACTION_DELTAS[action]
        candidate = (pos[0] + dr, pos[1] + dc)

        info = {"bumped_wall": False, "bumped_obstacle": False}

        if not self.in_bounds(candidate):
            info["bumped_wall"] = True       # tried to leave the grid
            return pos, info                 # robot stays where it is
        if self.is_obstacle(candidate):
            info["bumped_obstacle"] = True   # tried to enter an obstacle
            return pos, info                 # robot stays where it is
        return candidate, info               # legal move

    def step(self, action):
        """
        Apply one action. Returns the standard RL 4-tuple:

            new_pos : the robot's position after the action
            reward  : float, see reward() below
            done    : True when the episode is over (goal reached, or
                      the step limit was hit)
            info    : dict with bumped_wall / bumped_obstacle / reached_goal
        """
        new_pos, info = self.next_position(self.robot_pos, action)
        self.robot_pos = new_pos
        self.step_count += 1

        info["reached_goal"] = (new_pos == self.goal)
        reward = self.reward(info)
        done = info["reached_goal"] or self.step_count >= self.max_steps
        return new_pos, reward, done, info

    @staticmethod
    def reward(info):
        """
        Map what happened in a step to a number.

        Three mutually exclusive cases, checked from best to worst:
          reached the goal          -> REWARD_GOAL       (+100)
          bumped a wall / obstacle  -> REWARD_COLLISION  (-10)
          ordinary move             -> REWARD_STEP       (-1)
        """
        if info["reached_goal"]:
            return REWARD_GOAL
        if info["bumped_wall"] or info["bumped_obstacle"]:
            return REWARD_COLLISION
        return REWARD_STEP

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------
    def render(self, title="Grid World", save_path=None, show=True,
               path=None, policy=None, ax=None):
        """
        Draw the grid with matplotlib.

        white  = free cell        dark grey = obstacle
        S (blue) = start          G (green) = goal
        red circle = robot's current position

        Phase 5 extras:
          path   - list of (row, col) positions; drawn as an orange line
          policy - (size, size) array of actions; drawn as a small arrow
                   in every free cell (the agent's greedy choice there)
          ax     - draw into an existing axes (for side-by-side figures)
        """
        own_figure = ax is None
        if own_figure:
            fig, ax = plt.subplots(figsize=(6, 6))
        else:
            fig = ax.figure

        # Draw the map. imshow treats array index [row, col] as [y, x],
        # which matches our (row, col) convention with row 0 on top.
        cmap = ListedColormap(["white", "dimgray"])
        ax.imshow(self.grid, cmap=cmap, vmin=0, vmax=1)

        # Thin grid lines between cells.
        ax.set_xticks(np.arange(-0.5, self.size, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, self.size, 1), minor=True)
        ax.grid(which="minor", color="lightgray", linewidth=1)
        ax.tick_params(which="minor", length=0)

        # Major ticks show the row / col numbers.
        ax.set_xticks(range(self.size))
        ax.set_yticks(range(self.size))
        ax.set_xlabel("col")
        ax.set_ylabel("row")

        # Policy arrows: one per free cell, pointing in the greedy direction.
        if policy is not None:
            for r in range(self.size):
                for c in range(self.size):
                    if self.grid[r, c] == OBSTACLE or (r, c) == self.goal:
                        continue
                    dr, dc = ACTION_DELTAS[int(policy[r, c])]
                    ax.annotate("", xy=(c + 0.35 * dc, r + 0.35 * dr),
                                xytext=(c - 0.35 * dc, r - 0.35 * dr),
                                arrowprops=dict(arrowstyle="->", lw=1.2,
                                                color="steelblue"))

        # Learned / scripted path as a line through cell centres.
        if path:
            rows = [p[0] for p in path]
            cols = [p[1] for p in path]
            ax.plot(cols, rows, "-", color="darkorange", linewidth=3, alpha=0.9)
            ax.plot(cols, rows, "o", color="darkorange", markersize=5)

        # Robot marker (drawn first so the S/G labels stay readable on top).
        rr, rc = self.robot_pos
        ax.plot(rc, rr, "o", markersize=18, color="crimson",
                markeredgecolor="black", alpha=0.6)

        # Start and goal labels. Note: ax.text takes (x, y) = (col, row).
        sr, sc = self.start
        gr, gc = self.goal
        ax.text(sc, sr, "S", ha="center", va="center",
                fontsize=16, fontweight="bold", color="royalblue")
        ax.text(gc, gr, "G", ha="center", va="center",
                fontsize=16, fontweight="bold", color="seagreen")

        ax.set_title(title)
        fig.tight_layout()

        if save_path is not None:
            fig.savefig(save_path, dpi=120)
        if show:
            plt.show()
        elif own_figure:
            plt.close(fig)
        return fig, ax


if __name__ == "__main__":
    # Quick manual check: python -m env.grid_env   (run from project root)
    env = GridEnv()
    print(env.grid)
    env.render(save_path="results/phase1_grid.png")
