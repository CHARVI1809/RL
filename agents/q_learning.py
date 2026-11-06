"""
q_learning.py  --  Phase 4: tabular Q-learning.

The agent keeps a table Q[row, col, action] = "how good is it to take
`action` from cell (row, col)?"  Every step it nudges one entry of the
table toward a better estimate using the reward it just received.
"""

import numpy as np

from env.grid_env import ACTIONS


class QLearningAgent:
    def __init__(self, grid_size, n_actions=len(ACTIONS),
                 alpha=0.1, gamma=0.95,
                 epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.995,
                 seed=0):
        # The Q-table. One row per grid cell, one column per action.
        # Starts at zero: the agent initially thinks every action is equal.
        self.Q = np.zeros((grid_size, grid_size, n_actions))

        self.alpha = alpha                  # learning rate
        self.gamma = gamma                  # discount factor
        self.epsilon = epsilon              # exploration rate (current)
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.n_actions = n_actions
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Acting
    # ------------------------------------------------------------------
    def greedy_action(self, state):
        """Best known action for `state`, breaking ties at random.

        Random tie-breaking matters: the table starts all-zero, and a plain
        np.argmax would always return action 0 (UP) - the robot would spend
        its first episodes bumping into the top wall."""
        q_values = self.Q[state]
        best = np.flatnonzero(q_values == q_values.max())
        return int(self.rng.choice(best))

    def act(self, state):
        """Final-policy interface shared with the baselines: greedy, no exploration."""
        return self.greedy_action(state)

    def reset(self):
        """Nothing to reset between evaluation episodes (the table is the policy)."""

    def choose_action(self, state):
        """Epsilon-greedy: explore with probability epsilon, else exploit."""
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_actions))   # explore
        return self.greedy_action(state)                     # exploit

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------
    def update(self, state, action, reward, next_state, terminal):
        """
        One Q-learning update:

            target = reward                                 if terminal
                   = reward + gamma * max_a Q[next_state]   otherwise
            Q[state, action] += alpha * (target - Q[state, action])

        `terminal` means "the episode is truly over" (goal reached). If the
        episode merely hit the step limit, the next state still has value,
        so we keep bootstrapping from it.
        """
        current = self.Q[state][action]
        if terminal:
            target = reward
        else:
            target = reward + self.gamma * self.Q[next_state].max()
        self.Q[state][action] = current + self.alpha * (target - current)
        return target - current           # the TD error, handy for debugging

    def decay_epsilon(self):
        """Called once per episode: explore less as the agent learns more."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


# ----------------------------------------------------------------------
# Training loop
# ----------------------------------------------------------------------
def train(env, agent, n_episodes=1000, log_every=100, record_episodes=()):
    """
    Run `n_episodes` episodes of epsilon-greedy Q-learning.

    Returns a history dict with one entry per episode:
        reward   - total reward in the episode
        steps    - number of steps taken
        success  - True if the goal was reached
        epsilon  - exploration rate used for that episode
        collisions - wall + obstacle bumps in the episode
    plus
        paths    - {episode_number: [positions...]} for every episode listed
                   in `record_episodes` (1-based). Lets you animate what the
                   robot actually did early in training.
    """
    history = {"reward": [], "steps": [], "success": [], "epsilon": [],
               "collisions": [], "paths": {}}
    record_episodes = set(record_episodes)

    for ep in range(1, n_episodes + 1):
        state = env.reset()
        total_reward, done, collisions = 0.0, False, 0
        episode_path = [state]

        while not done:
            action = agent.choose_action(state)
            next_state, reward, done, info = env.step(action)
            agent.update(state, action, reward, next_state,
                         terminal=info["reached_goal"])
            state = next_state
            total_reward += reward
            collisions += info["bumped_wall"] or info["bumped_obstacle"]
            episode_path.append(state)

        history["reward"].append(total_reward)
        history["steps"].append(env.step_count)
        history["success"].append(info["reached_goal"])
        history["epsilon"].append(agent.epsilon)
        history["collisions"].append(collisions)
        if ep in record_episodes:
            history["paths"][ep] = episode_path
        agent.decay_epsilon()

        if log_every and ep % log_every == 0:
            recent = slice(-log_every, None)
            print(f"ep {ep:5d} | avg reward {np.mean(history['reward'][recent]):7.1f} "
                  f"| avg steps {np.mean(history['steps'][recent]):6.1f} "
                  f"| success {np.mean(history['success'][recent]):5.0%} "
                  f"| eps {agent.epsilon:.3f}")

    return history


def greedy_rollout(env, agent, max_steps=None):
    """Follow the learned policy with no exploration. Returns the path
    (list of positions, including the start) and whether the goal was hit."""
    max_steps = max_steps or env.max_steps
    state = env.reset()
    path = [state]
    for _ in range(max_steps):
        state, _, _, info = env.step(agent.greedy_action(state))
        path.append(state)
        if info["reached_goal"]:
            return path, True
    return path, False
