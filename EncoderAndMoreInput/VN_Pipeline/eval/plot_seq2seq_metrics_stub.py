"""
Stub for plotting Seq2SeqRewriter metrics.

Target graphs:
- Val loss vs step
- Output length ratio vs step
- Copy match rate vs step
- Exact match rate vs step (optional)

Output: PNG files saved to seq2seq_output_dir/plots/
"""

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


def load_metrics(csv_path: Path) -> dict[str, list]:
    """
    TODO: Load CSV rows into a structure you can plot.
    
    Steps:
    1. Open CSV file, read with csv.DictReader.
    2. Build a dict of lists: {"step": [...], "val_loss": [...], ...}
    3. Convert numeric strings to float/int.
    4. Handle missing values (empty cells) - use None or skip row.
    
    Expected CSV columns:
    step, timestamp, val_loss, avg_len_ratio, copy_match_rate, exact_match_rate
    """
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"Metrics CSV not found: {csv_path}")
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        metrics = {
            "step": [],
            "val_loss": [],
            "avg_len_ratio": [],
            "copy_match_rate": [],
            "exact_match_rate": [],
        }
        for row in reader:
            for key, value in row.items():
                if key not in metrics:
                    metrics[key] = []
                if value == "":
                    metrics[key].append(None)
                else:
                    metrics[key].append(float(value))
    return metrics


def plot_val_loss(metrics: dict[str, list], output_path: Path) -> None:
    """
    TODO: Plot validation loss vs training step.
    
    Steps:
    1. Extract metrics["step"] and metrics["val_loss"].
    2. Filter out None values.
    3. Create figure, plot, add labels/title/grid.
    4. Save to output_path.
    """
    steps = metrics["step"]
    losses = metrics["val_loss"]
    valid = [(s, l) for s, l in zip(steps, losses) if l is not None]
    if not valid:
        print("No val_loss data to plot")
        return
    steps, losses = zip(*valid)
    plt.figure(figsize=(10, 6))
    plt.plot(steps, losses, marker='o', linewidth=2, markersize=4)
    plt.xlabel("Training Step")
    plt.ylabel("Validation Loss")
    plt.title("Seq2Seq Validation Loss")
    plt.grid(True, alpha=0.3)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')


def plot_len_ratio(metrics: dict[str, list], output_path: Path) -> None:
    """
    TODO: Plot average length ratio vs training step.
    
    Steps:
    1. Extract metrics["step"] and metrics["avg_len_ratio"].
    2. Add reference line at y=1.0 (ideal ratio).
    3. Add shaded "acceptable" region (0.9 to 1.1).
    4. Save figure.
    """
    steps = metrics["step"]
    avg_len_ratio = metrics["avg_len_ratio"]
    valid = [(s, l) for s, l in zip(steps, avg_len_ratio) if l is not None]
    if not valid:
        print("No avg_len_ratio data to plot")
        return
    steps, avg_len_ratio = zip(*valid)
    ref_line = [1.0] * len(steps)
    acceptable_region = ([0.9] * len(steps), [1.1] * len(steps))
    plt.figure(figsize=(10, 6))
    plt.plot(steps, avg_len_ratio, marker='o', linewidth=2, markersize=4)
    plt.plot(steps, ref_line, linestyle='--', color='red', label='Ideal (1:1)')
    plt.fill_between(steps, acceptable_region[0], acceptable_region[1], alpha=0.2, color='green', label='Acceptable (0.9-1.1)')
    plt.xlabel("Training Step")
    plt.ylabel("Output/Input Length Ratio")
    plt.title("Seq2Seq Length Ratio")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')


def plot_copy_match(metrics: dict[str, list], output_path: Path) -> None:
    """
    TODO: Plot copy match rate vs training step.
    
    Steps:
    1. Extract metrics["step"] and metrics["copy_match_rate"].
    2. Y-axis should be 0-1 range.
    3. Add reference line at y=1.0 (perfect).
    4. Save figure.
    """
    steps = metrics["step"]
    copy_match_rate = metrics["copy_match_rate"]
    y_range = ([0.0] * len(steps), [1.0] * len(steps))
    y_ref = [1.0] * len(steps)
    plt.figure(figsize=(10, 6))
    plt.plot(steps, copy_match_rate, marker='o', linewidth=2, markersize=4)
    plt.fill_between(steps, y_range[0], y_range[1], alpha=0.2, color='green', label='Acceptable (0-1)')
    plt.plot(steps, y_ref, linestyle='--', color='red', label='Perfect (1)')
    plt.xlabel("Training Step")
    plt.ylabel("Copy Match Rate")
    plt.title("Seq2Seq Copy Match Rate")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')


def plot_all_metrics(metrics: dict[str, list], output_path: Path) -> None:
    """
    TODO: Create a combined figure with all metrics as subplots.
    
    Steps:
    1. Create 2x2 subplot figure.
    2. Plot each metric in its own subplot.
    3. Save as single image for quick overview.
    """
    subplots = [
        (plot_val_loss, metrics, "val_loss.png"),
        (plot_len_ratio, metrics, "len_ratio.png"),
        (plot_copy_match, metrics, "copy_match.png"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    def _filter_pairs(x_vals: list, y_vals: list) -> tuple[list, list]:
        xs = []
        ys = []
        for x, y in zip(x_vals, y_vals):
            if y is not None:
                xs.append(x)
                ys.append(y)
        return xs, ys

    axes_flat = axes.ravel()
    steps = metrics["step"]

    ax = axes_flat[0]
    xs, ys = _filter_pairs(steps, metrics["val_loss"])
    if xs:
        ax.plot(xs, ys, marker="o", linewidth=2, markersize=4)
        ax.set_xlabel("Training Step")
        ax.set_ylabel("Validation Loss")
        ax.set_title("Seq2Seq Validation Loss")
        ax.grid(True, alpha=0.3)
    else:
        ax.set_visible(False)

    ax = axes_flat[1]
    xs, ys = _filter_pairs(steps, metrics["avg_len_ratio"])
    if xs:
        ax.plot(xs, ys, marker="o", linewidth=2, markersize=4)
        ax.plot(xs, [1.0] * len(xs), linestyle="--", color="red", label="Ideal (1:1)")
        ax.fill_between(xs, [0.9] * len(xs), [1.1] * len(xs), alpha=0.2, color="green", label="Acceptable (0.9-1.1)")
        ax.set_xlabel("Training Step")
        ax.set_ylabel("Output/Input Length Ratio")
        ax.set_title("Seq2Seq Length Ratio")
        ax.grid(True, alpha=0.3)
        ax.legend()
    else:
        ax.set_visible(False)

    ax = axes_flat[2]
    xs, ys = _filter_pairs(steps, metrics["copy_match_rate"])
    if xs:
        ax.plot(xs, ys, marker="o", linewidth=2, markersize=4)
        ax.fill_between(xs, [0.0] * len(xs), [1.0] * len(xs), alpha=0.2, color="green", label="Acceptable (0-1)")
        ax.plot(xs, [1.0] * len(xs), linestyle="--", color="red", label="Perfect (1)")
        ax.set_xlabel("Training Step")
        ax.set_ylabel("Copy Match Rate")
        ax.set_title("Seq2Seq Copy Match Rate")
        ax.grid(True, alpha=0.3)
        ax.legend()
    else:
        ax.set_visible(False)

    ax = axes_flat[3]
    xs, ys = _filter_pairs(steps, metrics.get("exact_match_rate", []))
    if xs:
        ax.plot(xs, ys, marker="o", linewidth=2, markersize=4)
        ax.plot(xs, [1.0] * len(xs), linestyle="--", color="red", label="Perfect (1)")
        ax.set_xlabel("Training Step")
        ax.set_ylabel("Exact Match Rate")
        ax.set_title("Seq2Seq Exact Match Rate")
        ax.grid(True, alpha=0.3)
        ax.legend()
    else:
        ax.set_visible(False)
    fig.suptitle("Seq2Seq Training Metrics", fontsize=14)
    plt.tight_layout()
    combined_output_path = output_path
    plt.savefig(combined_output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {combined_output_path}")


def main() -> None:
    """
    CLI entry point.
    
    Steps:
    1. Parse args: --csv, --output-dir.
    2. Load config.json to get default paths if not specified.
    3. Load metrics via load_metrics().
    4. Call each plot function.
    """
    config_path = Path("EncoderAndMoreInput/VN_Pipeline/config.json")
    
    # Load config first (if it exists) to get default paths
    config = {}
    if config_path.exists():
        with open(config_path, "r") as f:
            config = json.load(f)
    
    # Determine paths from config or use defaults
    seq2seq_dir = Path(config.get("seq2seq_output_dir", "EncoderAndMoreInput/VN_Pipeline/out/seq2seq"))
    csv_file = seq2seq_dir / "metrics.csv"
    output_dir = seq2seq_dir / "plots"
    
    # Check CSV exists before loading
    if not csv_file.is_file():
        raise FileNotFoundError(f"Metrics CSV not found: {csv_file}")
    
    # Create output directory if needed
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load metrics and generate plots
    metrics = load_metrics(csv_file)
    
    plot_val_loss(metrics, output_dir / "val_loss.png")
    plot_len_ratio(metrics, output_dir / "len_ratio.png")
    plot_copy_match(metrics, output_dir / "copy_match.png")
    plot_all_metrics(metrics, output_dir / "all_metrics.png")



if __name__ == "__main__":
    main()
