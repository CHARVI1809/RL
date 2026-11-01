"""
plotting.py  --  Phase 5: training curves.

The raw per-episode reward is very noisy (epsilon-greedy adds randomness
every episode), so each curve is shown twice: faint raw values and a
bold moving average that makes the trend readable.
"""

import numpy as np
import matplotlib.pyplot as plt


def moving_average(values, window=50):
    """Average of each value with the (window-1) values before it.
    Shorter at the start so the output has the same length as the input."""
    values = np.asarray(values, dtype=float)
    out = np.empty_like(values)
    cumsum = np.cumsum(np.insert(values, 0, 0.0))
    for i in range(len(values)):
        lo = max(0, i - window + 1)
        out[i] = (cumsum[i + 1] - cumsum[lo]) / (i + 1 - lo)
    return out


def plot_training(history, window=50, save_path=None, show=True):
    """Three panels: total reward, steps per episode, success rate."""
    episodes = np.arange(1, len(history["reward"]) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    panels = [
        ("reward",  "Total reward per episode", "reward"),
        ("steps",   "Steps per episode",        "steps"),
        ("success", f"Success rate (window={window})", "fraction"),
    ]
    for ax, (key, title, ylabel) in zip(axes, panels):
        raw = np.asarray(history[key], dtype=float)
        if key != "success":                      # success is 0/1: avg only
            ax.plot(episodes, raw, color="lightgray", linewidth=0.8, label="raw")
        ax.plot(episodes, moving_average(raw, window), color="tab:blue",
                linewidth=2, label=f"{window}-ep average")
        ax.set_title(title)
        ax.set_xlabel("episode")
        ax.set_ylabel(ylabel)
        ax.legend(loc="best")

    # Epsilon on a second axis of the first panel: shows explore -> exploit.
    eps_ax = axes[0].twinx()
    eps_ax.plot(episodes, history["epsilon"], color="tab:red",
                linewidth=1, linestyle="--", label="epsilon")
    eps_ax.set_ylabel("epsilon", color="tab:red")
    eps_ax.set_ylim(0, 1.05)

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=120)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig
