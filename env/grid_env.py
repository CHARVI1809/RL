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


class GridEnv:
    """A static grid world with a start cell, a goal cell and obstacles."""

    def __init__(self, size=10, start=(0, 0), goal=(9, 9), obstacles=None):
        self.size = size
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
        """Put the robot back on the start cell and return that position."""
        self.robot_pos = self.start
        return self.robot_pos

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------
    def render(self, title="Grid World", save_path=None, show=True):
        """
        Draw the grid with matplotlib.

        white  = free cell        dark grey = obstacle
        S (blue) = start          G (green) = goal
        red circle = robot's current position
        """
        fig, ax = plt.subplots(figsize=(6, 6))

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
        else:
            plt.close(fig)
        return fig, ax


if __name__ == "__main__":
    # Quick manual check: python -m env.grid_env   (run from project root)
    env = GridEnv()
    print(env.grid)
    env.render(save_path="results/phase1_grid.png")
