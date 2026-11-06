"""
evaluation.py  --  Phase 6: measure an agent and compare agents.

Metrics per agent (averaged over n_episodes greedy episodes):
    success_rate   fraction of episodes that reached the goal
    avg_steps      mean steps per episode (failures count max_steps)
    avg_reward     mean total reward per episode
    avg_collisions mean number of wall + obstacle bumps per episode

Plus, for the learner only:
    convergence_episode   from the training history (see below)
"""

import numpy as np

from utils.plotting import moving_average


def run_episode(env, agent, record_path=False):
    """One episode following agent.act() with no exploration.
    With record_path=True the dict also holds "path": every position visited."""
    state = env.reset()
    agent.reset()
    total, collisions, done = 0.0, 0, False
    path = [state]
    while not done:
        state, reward, done, info = env.step(agent.act(state))
        total += reward
        collisions += info["bumped_wall"] or info["bumped_obstacle"]
        path.append(state)
    out = {"success": info["reached_goal"], "steps": env.step_count,
           "reward": total, "collisions": collisions}
    if record_path:
        out["path"] = path
    return out


def evaluate(env, agent, n_episodes=100):
    """Average the per-episode metrics over n_episodes."""
    runs = [run_episode(env, agent) for _ in range(n_episodes)]
    return {
        "success_rate":   float(np.mean([r["success"] for r in runs])),
        "avg_steps":      float(np.mean([r["steps"] for r in runs])),
        "avg_reward":     float(np.mean([r["reward"] for r in runs])),
        "avg_collisions": float(np.mean([r["collisions"] for r in runs])),
        "n_episodes":     n_episodes,
    }


def convergence_episode(history, window=50, tol=5.0):
    """
    The episode after which the smoothed reward never again strays more
    than `tol` from its final value. In other words: when did the curve
    flatten out for good?

    Returns the 1-based episode number.
    """
    ma = moving_average(history["reward"], window)
    final = ma[-1]
    off = np.flatnonzero(np.abs(ma - final) > tol)
    return int(off[-1]) + 2 if len(off) else 1   # +1 for index, +1 for "after"


def compare(env, agents, n_episodes=100):
    """agents: {name: agent}. Returns {name: metrics}."""
    return {name: evaluate(env, agent, n_episodes) for name, agent in agents.items()}


def format_table(results):
    """Markdown table of compare() output, ready for a README or report."""
    cols = [("success_rate", "Success", "{:.0%}"),
            ("avg_steps", "Avg steps", "{:.1f}"),
            ("avg_reward", "Avg reward", "{:.1f}"),
            ("avg_collisions", "Avg collisions", "{:.1f}")]
    lines = ["| Agent | " + " | ".join(c[1] for c in cols) + " |",
             "|---|" + "---|" * len(cols)]
    for name, m in results.items():
        lines.append(f"| {name} | " + " | ".join(f.format(m[k]) for k, _, f in cols) + " |")
    return "\n".join(lines)
