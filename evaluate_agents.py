"""
Phase 6: evaluate the trained Q-learning agent against a random agent and A*.

    python evaluate_agents.py

Writes results/comparison.png, results/comparison.md and
results/agents_side_by_side.gif (one episode of each agent, animated).
"""

import numpy as np

from env.grid_env import GridEnv
from agents.q_learning import QLearningAgent, train
from agents.baselines import RandomAgent, AStarAgent
from evaluation import compare, convergence_episode, format_table
from utils.plotting import plot_comparison
from utils.animate import animate_agents
from evaluation import run_episode


def main(n_train=1000, n_eval=100, seed=0, show=False):
    env = GridEnv()

    # 1. Train the learner (same settings as Phase 5).
    q_agent = QLearningAgent(grid_size=env.size, seed=seed)
    history = train(env, q_agent, n_episodes=n_train, log_every=0)
    conv = convergence_episode(history)

    # 2. Baselines.
    agents = {
        "Random": RandomAgent(seed=seed),
        "Q-learning": q_agent,
        "A*": AStarAgent(env),
    }

    # 3. Evaluate every agent on the same environment.
    results = compare(env, agents, n_episodes=n_eval)
    table = format_table(results)

    # 4. Training-time metrics for the learner.
    total_training_collisions = int(np.sum(history["collisions"]))
    first_100_coll = float(np.mean(history["collisions"][:100]))
    last_100_coll = float(np.mean(history["collisions"][-100:]))

    report = "\n".join([
        f"## Evaluation ({n_eval} greedy episodes per agent, seed={seed})", "",
        table, "",
        "## Q-learning training convergence",
        f"- Episodes trained: {n_train}",
        f"- Reward curve flattened (50-ep average within 5 of final) at episode: {conv}",
        f"- Collisions per episode: first 100 eps = {first_100_coll:.1f}, "
        f"last 100 eps = {last_100_coll:.1f}, total over training = {total_training_collisions}",
        f"- A* optimal path length: {len(agents['A*'].actions)} steps",
    ])
    print(report)

    with open("results/comparison.md", "w") as f:
        f.write(report + "\n")
    plot_comparison(results, save_path="results/comparison.png", show=show)

    # 5. Watch one episode of each agent side by side.
    paths = {name: run_episode(env, ag, record_path=True)["path"]
             for name, ag in agents.items()}
    animate_agents(env, paths, save_path="results/agents_side_by_side.gif")
    return results, history, conv


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    main()
