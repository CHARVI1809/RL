# Test log

Every phase has its own `test_phaseN.py`. Each file is a plain script with
`test_*` functions and `assert` statements - no pytest needed. Run one phase
with `python test_phaseN.py`, or all of them in order with
`python run_tests.py` (stops at the first failure so the traceback points at
the right phase). Always run from the project root.

The tests double as the project's build history: each one records a property
that was decided during that phase and must stay true afterwards. When a
later phase changes something (e.g. Phase 3 changed `step()` to return four
values), the earlier tests are updated and re-run, never deleted.

## Phase 1 - Environment (8 tests)

| Test | Property it protects |
|------|----------------------|
| `test_grid_shape` | Map is a 10x10 integer array |
| `test_obstacle_count` | Every cell in `DEFAULT_OBSTACLES` is marked, nothing else is |
| `test_start_and_goal_are_free` | Start and goal are free cells and differ |
| `test_bounds_checks` | `in_bounds` / `is_obstacle` / `is_free` behave at edges, corners, negatives |
| `test_reset_puts_robot_on_start` | `reset()` restores the start position |
| `test_invalid_configs_are_rejected` | Start on obstacle, goal/obstacle out of bounds raise `ValueError` |
| `test_a_path_exists` | BFS proves the default map is solvable |
| `test_render_saves_image` | `render(save_path=...)` writes a non-empty PNG |

## Phase 2 - Movement (9 tests)

| Test | Property it protects |
|------|----------------------|
| `test_each_action_moves_one_cell` | UP/DOWN/LEFT/RIGHT move one cell in the right direction |
| `test_cannot_leave_grid` | All four edges block, flagged `bumped_wall` |
| `test_corners` | Corner cells block both outward directions |
| `test_obstacle_blocks_movement` | Obstacles block, flagged `bumped_obstacle` (not wall) |
| `test_goal_detection` | `reached_goal` only when landing on the goal |
| `test_invalid_action_rejected` | Actions outside 0..3 raise `ValueError` |
| `test_next_position_does_not_mutate_env` | `next_position` is a pure function |
| `test_all_cells_all_actions` | Exhaustive sweep: 84 free cells x 4 actions = 336 cases, every result legal |
| `test_scripted_route_reaches_goal` | 9x RIGHT + 9x DOWN reaches the goal with zero bumps |

## Phase 3 - Rewards (8 tests)

| Test | Property it protects |
|------|----------------------|
| `test_ordinary_move_reward` | Normal move = -1, not done |
| `test_wall_bump_reward` | Wall bump = -10, robot stays |
| `test_obstacle_bump_reward` | Obstacle bump = -10, right flag set |
| `test_goal_reward_and_done` | Goal = +100 and `done` |
| `test_reset_clears_step_count` | Step counter resets with the episode |
| `test_step_limit_ends_episode` | `done` exactly at `max_steps`, not before |
| `test_scripted_route_total_reward` | 18-move route totals exactly 83 |
| `test_reward_values_are_consistent` | `collision < step < 0 < goal`, and goal outweighs the BFS shortest path (18) |

## Phase 4 - Q-learning (8 tests)

| Test | Property it protects |
|------|----------------------|
| `test_q_table_shape_and_init` | Q is `(10, 10, 4)` and all zero |
| `test_greedy_picks_best_action` | epsilon = 0 returns the argmax action |
| `test_greedy_breaks_ties_randomly` | All-zero table yields all four actions, not just UP |
| `test_full_exploration_is_uniform_ish` | epsilon = 1 ignores the table (roughly uniform over 4000 draws) |
| `test_update_math_non_terminal` | Hand-checked: alpha=.5, gamma=.9, r=-1, maxQ'=10 gives Q=4.0 |
| `test_update_math_terminal` | Terminal update ignores Q(next) |
| `test_epsilon_decays_to_floor` | Geometric decay, floors at `epsilon_min` |
| `test_training_runs_and_learns` | 600 episodes: finite Q, late reward > early + 50, greedy rollout = 18 steps |

Observed in the training test (seed 42): first-50 avg reward -466.6,
last-50 avg reward 79.8, greedy path 18 steps.

## Phase 5 - Training and visualization (8 tests)

| Test | Property it protects |
|------|----------------------|
| `test_moving_average_math` | Hand-checked window-3 average; output length equals input; window 1 = raw |
| `test_plot_training_writes_file` | `plot_training` writes a PNG from a tiny fake history |
| `test_render_with_path_and_policy` | `render(path=..., policy=...)` draws without error |
| `test_render_into_existing_axes_does_not_close_it` | Drawing into a supplied `ax` leaves the figure open |
| `test_learned_path_is_valid` | Greedy route starts at S, ends at G, each step adjacent and free, no repeated cells |
| `test_animation_writes_gif_and_resets_env` | `animate_path` writes a GIF and leaves the robot back on the start cell |
| `test_recorded_episode_paths` | `train(record_episodes=...)` returns one legal position per step for exactly those episodes |
| `test_animation_max_frames_truncates` | `max_frames` caps the number of animation frames |

Observed from `python train_and_visualize.py` (seed 0, 1000 episodes):
last-100 success 100 %, avg steps 18.8, avg reward 80.5; greedy rollout
18 steps. Plots in `results/training_curves.png` and
`results/learned_path.png`.

## Phase 6 - Evaluation (9 tests)

| Test | Property it protects |
|------|----------------------|
| `test_astar_finds_shortest_path` | A* returns an 18-step S-to-G path and replaying its actions in the env reproduces it with no bumps |
| `test_astar_and_uniform_cost_agree` | Zero heuristic (BFS/UCS) gives the same optimal length |
| `test_astar_unreachable_goal` | Boxed-in start: `plan_path` returns `(None, None)`, `AStarAgent` raises |
| `test_run_episode_astar_metrics` | One A* episode = success, 18 steps, reward 83, 0 collisions |
| `test_evaluate_random_agent_is_sane` | Random agent metrics are in range and it does collide |
| `test_trained_agent_matches_astar` | Trained Q-agent: 100 % success, 18 steps, 0 collisions, same reward as A*; training collisions fall |
| `test_convergence_episode_math` | Hand-checked step-change history converges at episode 11; a flat history at 1 |
| `test_compare_table_and_plot` | `format_table` produces a 3-row markdown table; `plot_comparison` writes a PNG |
| `test_run_episode_records_path_and_side_by_side_gif` | `record_path=True` returns one position per step; `animate_agents` writes a GIF with the capped frame count and resets the env |

Observed from `python evaluate_agents.py` (seed 0): Random 17 % success /
188 steps / 53.5 collisions; Q-learning and A* both 100 % / 18 steps /
reward 83 / 0 collisions; convergence episode 473.

## Phase 7 - Moving obstacles (9 tests)

| Test | Property it protects |
|------|----------------------|
| `test_moving_obstacle_cycles` | Patrol cycles with the right period; non-adjacent paths rejected |
| `test_dynamic_env_state_and_validation` | State shape (10,10,4), reset returns (0,0,0), mover cells per time; paths on walls or the goal rejected |
| `test_safe_pass_and_walk_into_mover` | Moving past a departing mover is fine; moving into its next cell is a collision; clock advances |
| `test_swap_is_a_collision` | Robot and mover exchanging cells counts as a hit |
| `test_mover_runs_into_waiting_robot` | A mover entering the robot's cell while it waits is a hit |
| `test_waiting_is_safe_when_no_mover_arrives` | A wall bump with no mover arriving is only a wall bump |
| `test_render_with_movers` | Rendering with movers writes a PNG |
| `test_agent_with_phase_state` | Q-table takes the env's state shape; a 3-tuple state update changes exactly one entry |
| `test_learns_to_avoid_patrols_and_static_astar_does_not` | 2000 episodes: 100 % success, 0 collisions, <=20 steps; static A* collides at least once |

Observed from `python train_dynamic.py` (seed 0): Q-learning 100 % / 18 steps /
0 collisions; static A* 100 % / 19 steps / 1.0 collisions; Random 18 % /
57.6 collisions. Convergence episode 1167 of 3000.

## Running on your machine

Record your own numbers here after each phase so you can tell a real
regression from run-to-run noise:

| Date | Phase | Command | Result |
|------|-------|---------|--------|
| | | | |
