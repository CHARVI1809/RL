"""
Phase 5: train the Q-learning agent, plot the curves, draw the learned path.

    python train_and_visualize.py

Writes results/training_curves.png, results/learned_path.png,
results/robot_path.gif (the trained robot) and results/episode_NNN.gif
(the same robot early in training).
In a notebook, import the three steps below and call them with show=True.
"""

import numpy as np

from env.grid_env import GridEnv
from agents.q_learning import QLearningAgent, train, greedy_rollout
from utils.plotting import plot_training
from utils.animate import animate_path


def main(n_episodes=1000, seed=0, show=False):
    env = GridEnv()
    agent = QLearningAgent(grid_size=env.size, seed=seed)

    early = (1, 25, 100)                       # episodes worth watching
    history = train(env, agent, n_episodes=n_episodes, log_every=100,
                    record_episodes=early)
    plot_training(history, save_path="results/training_curves.png", show=show)

    # Greedy policy = the best action in every cell, and the route it yields.
    policy = agent.Q.argmax(axis=2)
    path, success = greedy_rollout(env, agent)
    env.render(title=f"Learned path: {len(path) - 1} steps, success={success}",
               path=path, policy=policy,
               save_path="results/learned_path.png", show=show)

    animate_path(env, path, save_path="results/robot_path.gif",
                 title="Trained robot (greedy)")

    # The same robot much earlier in training: mostly random wandering.
    for ep in early:
        ep_path = history["paths"][ep]
        animate_path(env, ep_path, save_path=f"results/episode_{ep:03d}.gif",
                     interval_ms=120, max_frames=80,
                     title=f"Episode {ep}, eps={history['epsilon'][ep - 1]:.2f}")
        print(f"episode {ep:3d}: {len(ep_path) - 1} steps, "
              f"reached goal = {ep_path[-1] == env.goal}")

    last = slice(-100, None)
    print("\nSummary of the last 100 training episodes:")
    print(f"  success rate : {np.mean(history['success'][last]):.0%}")
    print(f"  avg steps    : {np.mean(history['steps'][last]):.1f}")
    print(f"  avg reward   : {np.mean(history['reward'][last]):.1f}")
    print(f"Greedy rollout : {len(path) - 1} steps, reached goal = {success}")
    print("Route:", " -> ".join(str(p) for p in path))
    return history, path, success


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    main()
