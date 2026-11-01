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
