# RL-Based Robot Navigation

A simulated robot learns to navigate a 10×10 grid from a start cell to a goal
cell while avoiding obstacles, using tabular Q-learning. Pure Python + NumPy +
Matplotlib; runs comfortably on a CPU.

## Project status

| Phase | Topic | Status |
|-------|-------|--------|
| 1 | Environment setup (grid, start, goal, obstacles, visualization) | Done |
| 2 | Robot movement (4 actions, bounds and collision rules) | Done |
| 3 | Reward function | Done |
| 4 | Q-learning | Done |
| 5 | Training and visualization | Done |
| 6 | Performance evaluation (vs. random agent and BFS/A*) | In progress |
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
python test_phase3.py
python test_phase4.py
python test_phase5.py
python test_phase6.py
# or everything at once:
python run_tests.py
```

## Folder structure

```
rl_robot_navigation/
├── env/
│   └── grid_env.py      # GridEnv: the world the agent lives in
├── agents/
│   ├── q_learning.py    # QLearningAgent, train(), greedy_rollout()
│   └── baselines.py     # RandomAgent, A* planner, AStarAgent
├── utils/
│   ├── plotting.py      # moving_average(), plot_training()
│   └── animate.py       # animate_path(): GIF of the robot walking a route
├── notebooks/
│   └── phase5_watch_the_robot.ipynb   # train, plot, animate (GIF + inline player)
├── results/             # saved plots and Q-tables
├── test_phase1.py
├── test_phase2.py
├── test_phase3.py
├── test_phase4.py
├── test_phase5.py
├── test_phase6.py
├── run_tests.py         # runs every phase's tests in order
├── train_and_visualize.py  # Phase 5 entry point
├── evaluation.py        # run_episode, evaluate, compare, convergence_episode
├── evaluate_agents.py   # Phase 6 entry point
├── TESTS.md             # what every test checks, plus observed results
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

## Phase 3 — Reward function

`step(action)` now returns the standard RL 4-tuple `(new_pos, reward, done, info)`.

| Outcome | Reward | Reason |
|---|---|---|
| Reached the goal | +100 | Only positive reward; must outweigh the cost of walking there (shortest path = 18 steps → best episode scores 83). |
| Bumped a wall or obstacle | −10 | Must be worse than a normal step so going around is preferred over pushing in. |
| Ordinary move | −1 | Small cost per step makes shorter paths better. |

The cases are mutually exclusive (a bump means no movement, so no goal). No
distance-to-goal shaping: the step penalty plus goal reward already yield
shortest paths, and shaping can mislead the agent toward the wrong side of a
wall. `done` is also `True` after `max_steps` (default 200) as a safety net;
that is not penalised beyond the −1 per step already paid.

Constants live at the top of `grid_env.py`: `REWARD_GOAL`, `REWARD_STEP`,
`REWARD_COLLISION`, `DEFAULT_MAX_STEPS`. `reward(info)` is a static method so
the numbers can be changed in one place.

**Tests (8):** reward for each outcome, `done` on goal and on step limit,
`reset()` clears the step counter, the scripted route totals exactly 83, and
a consistency check (`collision < step < 0 < goal`, goal worth more than the
BFS shortest path).

## Phase 4 — Q-learning

`agents/q_learning.py` holds `QLearningAgent`, `train()` and `greedy_rollout()`.

The Q-table is `Q[row, col, action]`, shape `(10, 10, 4)`, initialised to
zero. Each step updates one entry:

```
target = reward                              if the goal was reached
       = reward + γ · max_a Q[next, a]       otherwise
Q[s, a] += α · (target − Q[s, a])
```

Defaults: α = 0.1, γ = 0.95, ε from 1.0 → 0.05 with ×0.995 decay per episode.

Two details that matter:

- **Random tie-breaking.** `greedy_action` picks randomly among all
  equal-best actions; plain `argmax` would always choose UP on a zero table.
- **`terminal` ≠ `done`.** The update treats only `reached_goal` as terminal.
  A step-limit ending leaves the robot on a cell that still has value, so
  bootstrapping continues from it.

`train(env, agent, n_episodes)` returns a history dict (`reward`, `steps`,
`success`, `epsilon` per episode). `greedy_rollout` follows the learned
policy with ε = 0 and returns the path.

Observed (seed 0, 1000 episodes, ~0.3 s on CPU): 100 % success from
episode ~200, average 18.8 steps late in training (ε = 0.05 adds roughly one
random move per episode), greedy rollout exactly 18 steps.

## Phase 5 — Training and visualization

`python train_and_visualize.py` trains for 1000 episodes and writes:

- `results/training_curves.png` — reward, steps and success rate per episode
  (raw + 50-episode moving average), with ε overlaid on the reward panel
- `results/learned_path.png` — the greedy route in orange plus the greedy
  action in every free cell as an arrow
- `results/robot_path.gif` — the trained robot walking its route
- `results/episode_001/025/100.gif` — the same robot early in training

`train(..., record_episodes=(1, 25, 100))` stores full position lists for the
named episodes in `history["paths"]`. `utils/animate.py::animate_path` turns
any path into a GIF (`max_frames` caps long early episodes). The notebook
`notebooks/phase5_watch_the_robot.ipynb` shows everything inline, including an
interactive player (`HTML(anim.to_jshtml())`).

Observed: success rate pins at 100 % around episode 100; steps keep falling to
~18.8 by episode 400 (ε = 0.05 adds ~1 random move per episode); greedy
rollout = 18 steps. Arrows in rarely visited pockets (e.g. (3,1), (1,6)) can
point in odd directions — Q-learning only guarantees good values for states
it visits often, and those cells are off every sensible route.

## Phase 6 — Evaluation

`python evaluate_agents.py` trains the learner, then runs 100 greedy episodes
for each of three agents on the same map and writes `results/comparison.md`
and `results/comparison.png`.

| Agent | What it is |
|---|---|
| Random | picks a uniformly random action each step (lower bound) |
| Q-learning | the trained table, ε = 0 (never saw the map) |
| A* | plans the optimal path from the full map, then replays it (upper bound). `plan_path` uses `env.next_position` as its neighbour function so it obeys the same movement rules as the robot; with a zero heuristic it is uniform-cost search / BFS |

Metrics (`evaluation.py`): success rate, average steps, average total reward,
average collisions (wall + obstacle bumps), and for the learner the
convergence episode — the point after which the 50-episode smoothed reward
never again leaves a ±5 band around its final value. `train()` now also
records collisions per episode.

Observed (seed 0):

| Agent | Success | Avg steps | Avg reward | Avg collisions |
|---|---|---|---|---|
| Random | 17% | 188.0 | -652.0 | 53.5 |
| Q-learning | 100% | 18.0 | 83.0 | 0.0 |
| A* | 100% | 18.0 | 83.0 | 0.0 |

Q-learning converged at episode 473; collisions per episode fell from 24.6
(first 100) to 0.2 (last 100). Its greedy policy is identical in length,
reward and collisions to A*'s optimal plan.

## Tests so far

- **Phase 1 (8 tests):** grid shape and dtype, obstacle placement, start/goal
  free, bounds helpers at edges and corners, `reset()`, invalid configs raise,
  a BFS proves the map is solvable, `render()` writes a PNG.
- **Phase 2 (9 tests):** each action moves one cell, all four edges and corners
  block, obstacles block with the right flag, goal detection, invalid actions
  raise, `next_position` is side-effect free, an exhaustive sweep of all
  84 free cells × 4 actions (336 cases), and a scripted 18-move route that
  reaches the goal with zero bumps.
- **Phase 3 (8 tests):** see the Phase 3 section.
- **Phase 4 (8 tests):** see the Phase 4 section.
- **Phase 5 (8 tests)** and **Phase 6 (8 tests):** see those sections.

Full per-test descriptions and observed numbers are in `TESTS.md`.
