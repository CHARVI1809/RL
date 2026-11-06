"""
animate.py  --  Phase 5 extra: watch the robot move.

Frame k draws the grid with the robot on path[k] and the trail it has
walked so far. Frames are stitched into a GIF with Pillow (already
installed as a matplotlib dependency), so nothing new is required.
"""

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


def animate_path(env, path, save_path="results/robot_path.gif",
                 interval_ms=250, policy=None, title="Robot navigation",
                 max_frames=None):
    """
    Build an animation of the robot following `path` (a list of (row, col)).

    max_frames: early-training episodes can be 200 steps long; pass e.g. 80
    to animate only the first 80 steps (the title says so).

    Returns the FuncAnimation object. In a notebook you can display it with
        from IPython.display import HTML; HTML(anim.to_jshtml())
    or simply open the saved GIF.
    """
    total = len(path) - 1
    if max_frames is not None and len(path) > max_frames:
        path = path[:max_frames]
        title = f"{title} (first {max_frames - 1} of {total} steps)"
    fig, ax = plt.subplots(figsize=(6, 6))

    def draw(k):
        ax.clear()
        env.robot_pos = path[k]                      # move the robot
        env.render(ax=ax, show=False, policy=policy,
                   path=path[:k + 1],                # trail so far
                   title=f"{title}  -  step {k}/{total}")
        return ax,

    anim = FuncAnimation(fig, draw, frames=len(path),
                         interval=interval_ms, repeat=True)
    if save_path:
        anim.save(save_path, writer=PillowWriter(fps=1000 / interval_ms))
    plt.close(fig)
    env.reset()                                      # leave the env tidy
    return anim


def animate_agents(env, paths, save_path="results/agents_side_by_side.gif",
                   interval_ms=200, max_frames=80):
    """
    Animate several agents on the same map side by side.

    paths: {name: [positions...]} - e.g. from evaluation.run_episode(..., record_path=True).
    Agents that finish early stay parked on their final cell while the
    others keep moving. max_frames caps the total length (random agents
    can take 200 steps).
    """
    names = list(paths)
    longest = max(len(p) for p in paths.values())
    n_frames = min(longest, max_frames) if max_frames else longest
    fig, axes = plt.subplots(1, len(names), figsize=(5 * len(names), 5))
    axes = [axes] if len(names) == 1 else list(axes)

    def draw(k):
        for ax, name in zip(axes, names):
            ax.clear()
            p = paths[name]
            j = min(k, len(p) - 1)                   # park finished agents
            env.robot_pos = p[j]
            status = "goal" if (j == len(p) - 1 and p[j] == env.goal) else f"{j}/{len(p) - 1}"
            env.render(ax=ax, show=False, path=p[:j + 1],
                       title=f"{name}  -  step {status}")
        return axes

    anim = FuncAnimation(fig, draw, frames=n_frames, interval=interval_ms, repeat=True)
    if save_path:
        anim.save(save_path, writer=PillowWriter(fps=1000 / interval_ms))
    plt.close(fig)
    env.reset()
    return anim
