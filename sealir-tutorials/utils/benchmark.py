"""
A timeit module that provides IPython-like %timeit functionality without IPython.
Built on top of Python's standard timeit module.
"""

import statistics
import timeit as _timeit
from typing import Callable, Optional


class TimeitResult:
    """Container for timing results with IPython-like formatting."""

    def __init__(self, times: list, loops: int, repeat: int):
        self.times = times
        self.loops = loops
        self.repeat = repeat
        self.best = min(times) / self.loops
        self.worst = max(times) / self.loops
        self.mean = statistics.mean(times) / self.loops
        self.stdev = (
            statistics.stdev(times) / self.loops if len(times) > 1 else 0
        )

    def __str__(self):
        # Format time with appropriate units
        def format_time(seconds):
            if seconds >= 1:
                return f"{seconds:.3f} s"
            elif seconds >= 1e-3:
                return f"{seconds*1e3:.3f} ms"
            elif seconds >= 1e-6:
                return f"{seconds*1e6:.3f} µs"
            else:
                return f"{seconds*1e9:.3f} ns"

        per_loop = self.best
        if self.repeat == 1:
            return f"{format_time(per_loop)} per loop (mean ± std. dev. of 1 run, {self.loops} loop{'s' if self.loops != 1 else ''} each)"
        else:
            return f"{format_time(per_loop)} per loop (mean ± std. dev. of {self.repeat} runs, {self.loops} loop{'s' if self.loops != 1 else ''} each)"

    def __repr__(self):
        return f"TimeitResult(best={self.best:.6f}s, mean={self.mean:.6f}s, loops={self.loops}, repeat={self.repeat})"


def timeit(
    func: Callable,
    repeat: int = 5,
    number: Optional[int] = None,
):
    # Create Timer object
    timer = _timeit.Timer(stmt=func)
    # Auto-determine number of loops if not specified
    if number is None:
        number = timer.autorange()[0]
    # Run repeat times and collect results
    times = timer.repeat(repeat=repeat, number=number)
    return TimeitResult(times, number, repeat)


def _create_matplotlib_plots(results, labels, title, figsize):
    """Create matplotlib plots for benchmark visualization."""
    import matplotlib.pyplot as plt
    import numpy as np

    # Create figure with subplots (now only 1 row, 2 columns)
    fig, (ax2, ax3) = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle(title, fontsize=16, fontweight="bold")

    # Extract data
    names = labels
    bests = [r.best for r in results]

    # 1. Box plot of all timing runs (ax2)
    cmap = plt.get_cmap("Set3")
    colors = cmap(np.linspace(0, 1, len(results)))
    all_times = []
    for result in results:
        per_loop_times = [t / result.loops for t in result.times]
        all_times.append(per_loop_times)

    bp = ax2.boxplot(all_times, labels=names, patch_artist=True)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax2.set_ylabel("Time per loop (seconds)")
    ax2.set_title("Distribution of Times")
    ax2.tick_params(axis="x", rotation=45)

    # 2. Speedup comparison (relative to first result) (ax3)
    if len(results) > 1:
        baseline = bests[0]
        speedups = [baseline / best for best in bests]
        bars = ax3.bar(
            names, speedups, color=colors, alpha=0.7, edgecolor="black"
        )
        ax3.set_ylabel("Speedup Factor")
        ax3.set_title("Relative Speedup")
        ax3.axhline(y=1.0, color="red", linestyle="--", alpha=0.5)
        ax3.tick_params(axis="x", rotation=45)

        # Add value labels
        for bar, speedup in zip(bars, speedups):
            height = bar.get_height()
            ax3.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{speedup:.2f}x",
                ha="center",
                va="bottom",
                fontsize=9,
            )
    else:
        ax3.text(
            0.5,
            0.5,
            "Need at least 2 results\nfor speedup comparison",
            ha="center",
            va="center",
            transform=ax3.transAxes,
            fontsize=12,
        )
        ax3.set_title("Relative Speedup")

    plt.tight_layout()
    plt.show()


def _print_benchmark_summary(results, labels):
    """Print benchmark summary without using matplotlib/numpy."""
    # Extract data
    names = labels
    bests = [r.best for r in results]

    # Print summary
    print("\n" + "=" * 50)
    print("BENCHMARK SUMMARY")
    print("=" * 50)

    if len(results) > 1:
        best_idx = min(range(len(bests)), key=lambda i: bests[i])
        print(f"Fastest: {names[best_idx]}")
        print(f"   Time: {bests[best_idx]:.2e} seconds per loop")

        print(f"\nSpeedup factors (vs {names[0]}):")
        baseline = bests[0]
        for i, (name, best) in enumerate(zip(names, bests)):
            speedup = baseline / best
            if speedup > 1:
                print(f"   {name}: {speedup:.2f}x faster")
            elif speedup < 1:
                print(f"   {name}: {1/speedup:.2f}x slower")
            else:
                print(f"   {name}: baseline")

    print(f"\nAll results:")
    for name, result in zip(names, results):
        print(f"   {name}: {result}")


def visualize_benchmark(
    *results, labels=None, title="Benchmark Comparison", figsize=(10, 6),
    show_plots=None,
):
    """
    Visualize benchmark results using matplotlib.

    Args:
        *results: TimeitResult objects to compare
        labels: List of labels for
        each result (optional, will use default names)
        title: Title for the plot
        figsize: Figure size tuple
        show_plots: Whether to display matplotlib plots.
                    If None, plots are shown only in notebook environments.

    Example:
        >>> result1 = timeit(func1)
        >>> result2 = timeit(func2)
        >>> visualize_benchmark(result1, result2,
                                labels=["Unoptimized", "Optimized"])
    """
    from . import IN_NOTEBOOK

    if not results:
        print("No results provided for visualization")
        return

    # Generate default labels if not provided
    if labels is None:
        labels = [f"Method {i+1}" for i in range(len(results))]
    elif len(labels) != len(results):
        print(
            f"Warning: {len(labels)} labels provided for {len(results)} results"
        )
        labels = labels[: len(results)] + [
            f"Method {i+1}" for i in range(len(labels), len(results))
        ]

    # Create plots only if in notebook environment
    if show_plots or (show_plots is None and IN_NOTEBOOK):
        _create_matplotlib_plots(results, labels, title, figsize)

    # Always print the summary (non-matplotlib functionality)
    _print_benchmark_summary(results, labels)