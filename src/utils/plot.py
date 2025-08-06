import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_categorical_distribution(
    labels: np.ndarray,
    label_name: str,
    title: str | None = None,
    use_latex: bool = True,
    rc_overrides: dict | None = None,
    figsize: tuple[float, float] = (8, 6),
    palette: str = "deep",
):
    """
    Plot a bar chart of the distribution of categorical labels.

    Args:
        labels         : 1D array of categorical labels (numeric or str).
        label_name     : name for the x‐axis (e.g. "Engagement Label").
        title          : plot title (defaults to f"Distribution of {label_name}s").
        use_latex      : if True, apply LaTeX rcParams (as in your script).
        rc_overrides   : extra rcParam overrides (merged on top of defaults).
        figsize        : figure size.
        palette        : seaborn palette for bars.
    """

    # 1) LaTeX settings
    if use_latex:
        defaults = {
            "text.usetex": True,
            "font.family": "serif",
            "axes.labelsize": 18,
            "axes.titlesize": 16,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "legend.fontsize": 16,
            "axes.labelweight": "bold",
            "axes.titleweight": "bold",
        }
        params = {**defaults, **(rc_overrides or {})}
        plt.rcParams.update(params)
    else:
        # Non-LaTeX settings
        defaults = {
            "text.usetex": False,
            "font.family": "serif",
            "axes.labelsize": 14,
            "axes.titlesize": 16,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
        }
        params = {**defaults, **(rc_overrides or {})}
        plt.rcParams.update(params)

    # 2) count unique labels
    unique_vals, counts = np.unique(labels, return_counts=True)
    x_ticks = [str(v) for v in unique_vals]

    # 3) draw
    plt.figure(figsize=figsize)
    sns.set_theme(style="whitegrid")
    bars = plt.bar(
        x_ticks,
        counts,
        color=sns.color_palette(palette)[0],
        edgecolor="black",
        width=0.7,
    )
    plt.xlabel(label_name, fontsize=14, fontweight="bold")
    plt.ylabel("Frequency", fontsize=14, fontweight="bold")
    plt.title(
        title or f"Distribution of {label_name}",
        fontsize=16,
        fontweight="bold",
        pad=15,
    )
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    # 4) annotate
    for bar, cnt in zip(bars, counts):
        h = bar.get_height()
        if h > 0:
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                h + 0.5,
                f"{int(cnt)}",
                ha="center",
                va="bottom",
                fontsize=11,
                fontweight="bold",
            )

    sns.despine()
    plt.tight_layout()
    plt.show()