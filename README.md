# RL-Based Robot Navigation

A simulated robot learns to navigate a 10×10 grid from a start cell to a goal
cell while avoiding obstacles, using tabular Q-learning. Pure Python + NumPy +
Matplotlib; runs comfortably on a CPU.

## Project status

| Phase | Topic | Status |
|-------|-------|--------|
| 1 | Environment setup (grid, start, goal, obstacles, visualization) | Done |
| 2 | Robot movement (4 actions, bounds and collision rules) | Done |
| 3 | Reward function | In progress |
| 4 | Q-learning | — |
| 5 | Training and visualization | — |
| 6 | Performance evaluation (vs. random agent and BFS/A*) | — |
| 7 | Enhancements | — |

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Run the tests from the project root:

```bash
python test_phase1.py
python test_phase2.py
```

## Folder structure

```
rl_robot_navigation/
├── env/
│   └── grid_env.py      # GridEnv: the world the agent lives in
├── agents/              # Q-learning, random and A* agents (later phases)
├── notebooks/           # Jupyter experiments; import from env/ and agents/
├── results/             # saved plots and Q-tables
├── test_phase1.py
├── test_phase2.py
├── requirements.txt
└── README.md
```

Logic lives in `.py` modules so notebooks stay short and experiments are
reproducible. On Google Colab, paste a module into a cell prefixed with
`%%writefile env/grid_env.py`.

## Conventions

- A position is a tuple `(row, col)`. **Row 0 is the top, col 0 is the left.**
  This matches NumPy indexing (`grid[row, col]`) and how `imshow` draws.
- The map is a NumPy integer array: `0 = FREE`, `1 = OBSTACLE`.
- The robot is *not* stored in the map; it lives in `env.robot_pos`.
- Actions are integers so they can index a Q-table later:
  `UP=0, DOWN=1, LEFT=2, RIGHT=3`.

## Phase 1 — Environment

`GridEnv(size=10, start=(0, 0), goal=(9, 9), obstacles=None)` builds the map
and validates it (obstacles must be in bounds; start and goal must be in
bounds and free). `DEFAULT_OBSTACLES` is a fixed 16-cell layout chosen so a
path from start to goal exists.

Helpers:

| Method | Purpose |
|--------|---------|
| `in_bounds(pos)` | Is `pos` inside the grid? |
| `is_obstacle(pos)` | Inside the grid *and* an obstacle? |
| `is_free(pos)` | Inside the grid *and* free? |
| `reset()` | Put the robot on the start cell; returns the position |
| `render(title, save_path, show)` | Draw the grid with Matplotlib |

Rendering: white = free, grey = obstacle, blue **S** = start, green **G** =
goal, red circle = robot. Note that Matplotlib takes `(x, y) = (col, row)`,
the reverse of our tuples; the code marks every place that swap happens.

## Phase 2 — Movement

`ACTION_DELTAS` maps each action to a `(Δrow, Δcol)` pair. `UP` is `(-1, 0)`
because rows grow downward.

Movement rule, applied in order:

1. Candidate cell outside the grid → robot stays put, `info["bumped_wall"] = True`
2. Candidate cell is an obstacle → robot stays put, `info["bumped_obstacle"] = True`
3. Otherwise → robot moves

"Stay put" (rather than ending the episode) keeps early training workable:
a random agent would otherwise die constantly and rarely see the goal.
A penalty for bumping is added in Phase 3.

| Method | Purpose |
|--------|---------|
| `next_position(pos, action)` | Pure function: `(new_pos, info)` without moving the robot. Reused by A* later. |
| `step(action)` | Moves the robot; adds `info["reached_goal"]`. Invalid actions raise `ValueError`. |

Quick manual check:

```python
from env.grid_env import GridEnv, RIGHT, DOWN, ACTION_NAMES
env = GridEnv(); env.reset()
for a in [RIGHT, RIGHT, DOWN]:
    print(ACTION_NAMES[a], env.step(a))
env.render()
```

The third step is refused (`bumped_obstacle: True`) because `(1, 2)` is a wall.

## Tests so far

- **Phase 1 (8 tests):** grid shape and dtype, obstacle placement, start/goal
  free, bounds helpers at edges and corners, `reset()`, invalid configs raise,
  a BFS proves the map is solvable, `render()` writes a PNG.
- **Phase 2 (9 tests):** each action moves one cell, all four edges and corners
  block, obstacles block with the right flag, goal detection, invalid actions
  raise, `next_position` is side-effect free, an exhaustive sweep of all
  84 free cells × 4 actions (336 cases), and a scripted 18-move route that
  reaches the goal with zero bumps.
