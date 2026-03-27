"""Direction A: Plot IB scaling laws — the hero figure.

Generates log-log plots of IB vs compute/parameters with loss scaling overlaid.
"""

import argparse
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Plot IB scaling laws")
    parser.add_argument("--results_file", type=str, required=True,
                        help="Path to scaling_laws.json from compute_scaling_laws.py")
    parser.add_argument("--output_dir", type=str, default=".",
                        help="Directory to save figures")
    parser.add_argument("--format", type=str, default="pdf", choices=["pdf", "png"])
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.results_file) as f:
        results = json.load(f)

    data = results["data_points"]
    labels = sorted(data.keys())
    n_params = np.array([data[l]["n_params"] for l in labels], dtype=float)
    losses = np.array([data[l]["loss"] for l in labels], dtype=float)
    ib_same = np.array([data[l]["ib_same_state"] for l in labels], dtype=float)
    ib_error = 1.0 - ib_same

    # ---- Figure 1: Loss vs Params and IB vs Params (log-log) ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Loss scaling
    ax = axes[0]
    ax.scatter(n_params, losses, c="tab:blue", s=60, zorder=5, label="NTP Loss")
    ls = results["loss_scaling"]
    if ls["alpha"] is not None:
        x_fit = np.linspace(n_params.min(), n_params.max(), 100)
        y_fit = ls["alpha"] * x_fit ** (-ls["beta"])
        ax.plot(x_fit, y_fit, "--", color="tab:blue", alpha=0.7,
                label=f"$L(N) \\propto N^{{-{ls['beta']:.2f}}}$ ($R^2$={ls['r_squared']:.2f})")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Parameters (N)", fontsize=12)
    ax.set_ylabel("Validation Loss", fontsize=12)
    ax.set_title("Loss Scaling", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # IB scaling
    ax = axes[1]
    ax.scatter(n_params, ib_error, c="tab:red", s=60, zorder=5,
               label="IB Error (1 - same_state_IB)")
    ibs = results["ib_scaling"]
    if ibs["alpha"] is not None:
        x_fit = np.linspace(n_params.min(), n_params.max(), 100)
        y_fit = ibs["alpha"] * x_fit ** (-ibs["beta"])
        ax.plot(x_fit, y_fit, "--", color="tab:red", alpha=0.7,
                label=f"$IB(N) \\propto N^{{-{ibs['beta']:.2f}}}$ ($R^2$={ibs['r_squared']:.2f})")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Parameters (N)", fontsize=12)
    ax.set_ylabel("IB Error", fontsize=12)
    ax.set_title("IB Scaling", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_dir / f"scaling_laws_separate.{args.format}",
                dpi=150, bbox_inches="tight")
    plt.close()

    # ---- Figure 2: Overlaid scaling (the hero figure) ----
    fig, ax = plt.subplots(figsize=(8, 6))

    # Normalize both to [0, 1] range for visual comparison
    loss_norm = (losses - losses.min()) / (losses.max() - losses.min() + 1e-10)
    ib_norm = (ib_error - ib_error.min()) / (ib_error.max() - ib_error.min() + 1e-10)

    ax.scatter(n_params, loss_norm, c="tab:blue", s=80, marker="o",
               zorder=5, label="NTP Loss (normalized)")
    ax.scatter(n_params, ib_norm, c="tab:red", s=80, marker="s",
               zorder=5, label="IB Error (normalized)")

    if ls["alpha"] is not None and ibs["alpha"] is not None:
        x_fit = np.linspace(n_params.min(), n_params.max(), 100)
        l_fit = ls["alpha"] * x_fit ** (-ls["beta"])
        l_fit_norm = (l_fit - losses.min()) / (losses.max() - losses.min() + 1e-10)
        ib_fit = ibs["alpha"] * x_fit ** (-ibs["beta"])
        ib_fit_norm = (ib_fit - ib_error.min()) / (ib_error.max() - ib_error.min() + 1e-10)
        ax.plot(x_fit, l_fit_norm, "--", color="tab:blue", alpha=0.7,
                label=f"Loss: $\\beta$ = {ls['beta']:.3f}")
        ax.plot(x_fit, ib_fit_norm, "--", color="tab:red", alpha=0.7,
                label=f"IB: $\\beta$ = {ibs['beta']:.3f}")

    ax.set_xscale("log")
    ax.set_xlabel("Parameters (N)", fontsize=14)
    ax.set_ylabel("Normalized Metric", fontsize=14)
    ax.set_title("Scaling Laws: Loss vs Inductive Bias", fontsize=15)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    if results.get("ib_gap") is not None:
        ax.text(0.05, 0.95,
                f"IB Gap ($|\\beta_L - \\beta_{{IB}}|$): {results['ib_gap']:.3f}",
                transform=ax.transAxes, fontsize=11, verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    plt.tight_layout()
    fig.savefig(output_dir / f"scaling_laws_hero.{args.format}",
                dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved figures to {output_dir}")


if __name__ == "__main__":
    main()
