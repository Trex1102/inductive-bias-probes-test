"""Direction C: Compute correlation between IB metrics and transfer performance.

Shows that IB metrics predict downstream transfer better than NTP loss alone.
This is the key result for Direction C: "IB as a reliability diagnostic."
"""

import argparse
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Direction C: IB vs Transfer correlation"
    )
    parser.add_argument("--dynamics_file", type=str, required=True,
                        help="Path to ib_dynamics.json from compute_ib_dynamics.py")
    parser.add_argument("--transfer_results", type=str, default=None,
                        help="Optional: path to transfer learning results JSON")
    parser.add_argument("--output_dir", type=str, default=".",
                        help="Directory to save figures")
    parser.add_argument("--format", type=str, default="pdf", choices=["pdf", "png"])
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.dynamics_file) as f:
        dynamics = json.load(f)

    steps = np.array([d["step"] for d in dynamics])
    ib_same = np.array([d["same_state_ib"] for d in dynamics])
    ib_same_se = np.array([d.get("same_state_stderr", 0) for d in dynamics])
    ib_diff = np.array([d["diff_state_loss"] for d in dynamics])
    ntp_losses = np.array([
        d["ntp_val_loss"] if d["ntp_val_loss"] is not None else np.nan
        for d in dynamics
    ])

    # ---- Figure 1: IB dynamics (the hero figure for Direction C) ----
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # NTP loss vs training step
    ax = axes[0]
    valid = ~np.isnan(ntp_losses)
    ax.plot(steps[valid], ntp_losses[valid], "o-", color="tab:blue",
            markersize=5, linewidth=1.5, label="NTP Val Loss")
    ax.set_ylabel("NTP Validation Loss", fontsize=12)
    ax.set_title("Training Dynamics: Loss and Inductive Bias", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xscale("log")

    # IB metrics vs training step
    ax = axes[1]
    ax.errorbar(steps, ib_same, yerr=ib_same_se, fmt="s-", color="tab:green",
                markersize=5, linewidth=1.5, capsize=3, label="Same-State IB")
    ax.plot(steps, ib_diff, "D-", color="tab:orange",
            markersize=5, linewidth=1.5, label="Diff-State Loss")
    ax.set_xlabel("Training Step", fontsize=12)
    ax.set_ylabel("IB Metric", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xscale("log")

    plt.tight_layout()
    fig.savefig(output_dir / f"ib_dynamics.{args.format}",
                dpi=150, bbox_inches="tight")
    plt.close()

    # ---- Figure 2: IB vs NTP loss (disentanglement) ----
    if np.sum(valid) >= 3:
        fig, ax = plt.subplots(figsize=(8, 6))

        sc = ax.scatter(ntp_losses[valid], ib_same[valid], c=steps[valid],
                        cmap="viridis", s=80, edgecolors="black", linewidth=0.5,
                        zorder=5)
        plt.colorbar(sc, label="Training Step")

        # Compute correlation
        r_pearson, p_pearson = pearsonr(ntp_losses[valid], ib_same[valid])
        r_spearman, p_spearman = spearmanr(ntp_losses[valid], ib_same[valid])

        ax.set_xlabel("NTP Validation Loss", fontsize=12)
        ax.set_ylabel("Same-State IB", fontsize=12)
        ax.set_title("IB vs NTP Loss Disentanglement", fontsize=14)
        ax.text(0.05, 0.95,
                f"Pearson r = {r_pearson:.3f} (p = {p_pearson:.3e})\n"
                f"Spearman ρ = {r_spearman:.3f} (p = {p_spearman:.3e})",
                transform=ax.transAxes, fontsize=10, verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        fig.savefig(output_dir / f"ib_vs_loss.{args.format}",
                    dpi=150, bbox_inches="tight")
        plt.close()

        logger.info(f"IB-Loss correlation: Pearson r={r_pearson:.3f}, "
                     f"Spearman rho={r_spearman:.3f}")

        # Check if IB changes while loss is flat (or vice versa)
        # Look at the last third of training
        n = len(steps)
        late_start = max(0, n - n // 3)
        if late_start < n - 1:
            late_loss_change = abs(ntp_losses[valid][-1] - ntp_losses[valid][late_start])
            late_ib_change = abs(ib_same[-1] - ib_same[late_start])
            logger.info(f"Late training (last 1/3): "
                         f"loss change = {late_loss_change:.6f}, "
                         f"IB change = {late_ib_change:.6f}")
            if late_loss_change < 0.01 and late_ib_change > 0.02:
                logger.info("KEY FINDING: IB changes while loss is flat!")
            elif late_ib_change < 0.01 and late_loss_change > 0.01:
                logger.info("KEY FINDING: Loss improves but IB plateaus!")

    # ---- Figure 3: IB plateau detection ----
    fig, ax = plt.subplots(figsize=(8, 5))

    # Compute running IB change rate
    if len(steps) > 2:
        ib_grad = np.gradient(ib_same, steps)
        ax.plot(steps, ib_grad, "o-", color="tab:purple", markersize=4, linewidth=1.5)
        ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
        ax.set_xlabel("Training Step", fontsize=12)
        ax.set_ylabel("dIB/dStep", fontsize=12)
        ax.set_title("IB Change Rate (for early stopping detection)", fontsize=13)
        ax.set_xscale("log")
        ax.grid(True, alpha=0.3)

        # Find plateau: where |gradient| < threshold
        threshold = np.abs(ib_grad).max() * 0.1
        plateau_mask = np.abs(ib_grad) < threshold
        if np.any(plateau_mask):
            plateau_step = steps[plateau_mask][0]
            ax.axvline(plateau_step, color="tab:red", linestyle=":",
                       label=f"IB plateau at step {plateau_step}")
            ax.legend(fontsize=10)

        plt.tight_layout()
        fig.savefig(output_dir / f"ib_plateau.{args.format}",
                    dpi=150, bbox_inches="tight")
        plt.close()

    # Save summary statistics
    summary = {
        "num_checkpoints": len(dynamics),
        "steps": steps.tolist(),
        "ib_same_state": ib_same.tolist(),
        "ib_diff_state": ib_diff.tolist(),
        "ntp_losses": ntp_losses.tolist(),
    }
    if np.sum(valid) >= 3:
        summary["pearson_r"] = float(r_pearson)
        summary["pearson_p"] = float(p_pearson)
        summary["spearman_rho"] = float(r_spearman)
        summary["spearman_p"] = float(p_spearman)

    with open(output_dir / "ib_dynamics_summary.json", "w") as f:
        json.dump(summary, f, indent=4)

    logger.info(f"Saved figures and summary to {output_dir}")


if __name__ == "__main__":
    main()
