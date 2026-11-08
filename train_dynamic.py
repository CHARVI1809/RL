"""
Phase 7: moving obstacles.

    python train_dynamic.py

Trains Q-learning on DynamicGridEnv (state = row, col, phase) and compares it
with the Phase 6 baselines, which plan on the static map and ignore the
patrols. Writes results/dynamic_training_curves.png, results/dynamic_path.png,
results/dynamic_comparison.md and results/dynamic_agents.gif.
"""

import numpy as np

from env.dynamic_grid_env import DynamicGridEnv
from agents.q_learning import QLearningAgent, train, greedy_rollout
from agents.baselines import RandomAgent, AStarAgent
from evaluation import compare, convergence_episode, format_table, run_episode
from utils.plotting import plot_training
from utils.animate import animate_agents


def main(n_train=3000, n_eval=100, seed=0, show=False):
    env = DynamicGridEnv()
    print(f"state shape {env.state_shape} -> {np.prod(env.state_shape)} states, "
          f"period {env.period}, movers {[m.path for m in env.movers]}")

    # Slower epsilon decay: 1600 states need more exploration than 100.
    agent = QLearningAgent(state_shape=env.state_shape, seed=seed, epsilon_decay=0.998)
    history = train(env, agent, n_episodes=n_train, log_every=500)
    conv = convergence_episode(history)
    plot_training(history, save_path="results/dynamic_training_curves.png", show=show)

    path, success = greedy_rollout(env, agent)
    print(f"\nGreedy rollout: {len(path) - 1} steps, reached goal = {success}, "
          f"converged at episode {conv}")

    # Snapshot of the learned route at t=0 (movers at their phase-0 cells).
    env.reset()
    env.render(title="Learned route", path=path,
               save_path="results/dynamic_path.png", show=show)

    agents = {
        "Random": RandomAgent(seed=seed),
        "Q-learning": agent,
        "A* (static map)": AStarAgent(env),     # plans with walls only
    }
    results = compare(env, agents, n_episodes=n_eval)
    table = format_table(results)
    report = "\n".join([
        f"## Moving obstacles: evaluation ({n_eval} greedy episodes, seed={seed})", "",
        table, "",
        f"- Q-learning converged at episode {conv} of {n_train}",
        f"- Static A* path length: {len(agents['A* (static map)'].actions)} steps "
        f"(ignores the patrols, so it collides)",
    ])
    print("\n" + report)
    with open("results/dynamic_comparison.md", "w") as f:
        f.write(report + "\n")

    paths = {n: run_episode(env, a, record_path=True)["path"] for n, a in agents.items()}
    animate_agents(env, paths, save_path="results/dynamic_agents.gif", max_frames=60)
    return results, history


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    main()
