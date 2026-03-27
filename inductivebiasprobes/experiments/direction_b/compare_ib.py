"""Direction B: Compare IB metrics between baseline and regularized models.

Loads IB results for baseline (standard NTP) and regularized (NTP + L_contrast)
models and generates comparison plots — the key figure for Direction B.
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
    parser = argparse.ArgumentParser(
        description="Direction B: Compare baseline vs regularized IB"
    )
    parser.add_argument("--baseline_ib", type=str, required=True,
                        help="Path to baseline ib.json")
    parser.add_argument("--regularized_ibs", type=str, nargs="+", required=True,
                        help="Paths to regularized ib.json files (one per lambda)")
    parser.add_argument("--lambda_values", type=float, nargs="+", required=True,
                        help="Lambda values corresponding to regularized_ibs")
    parser.add_argument("--output_dir", type=str, default=".",
                        help="Directory to save figures")
    parser.add_argument("--format", type=str, default="pdf", choices=["pdf", "png"])
    return parser.parse_args()


def load_ib(path):
    with open(path) as f:
        return json.load(f)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    assert len(args.regularized_ibs) == len(args.lambda_values), \
        "Must provide same number of regularized_ibs and lambda_values"

    baseline = load_ib(args.baseline_ib)
    regularized = []
    for path in args.regularized_ibs:
        regularized.append(load_ib(path))

    lambdas = args.lambda_values

    # ---- Figure 1: IB metrics vs lambda ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Same-state IB (higher is better)
    ax = axes[0]
    baseline_val = baseline["same_state_ib"]
    reg_vals = [r["same_state_ib"] for r in regularized]
    reg_errs = [r.get("same_state_stderr", 0) for r in regularized]

    ax.axhline(baseline_val, color="tab:gray", linestyle="--", linewidth=2,
               label=f"Baseline: {baseline_val:.3f}")
    ax.errorbar(lambdas, reg_vals, yerr=reg_errs, fmt="o-", color="tab:green",
                markersize=8, linewidth=2, capsize=4, label="Regularized")
    ax.set_xscale("log")
    ax.set_xlabel("$\\lambda$ (contrastive weight)", fontsize=12)
    ax.set_ylabel("Same-State IB (higher = better)", fontsize=12)
    ax.set_title("Same-State Inductive Bias", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Different-state loss (higher = better for world model)
    ax = axes[1]
    baseline_diff = baseline["diff_state_loss"]
    reg_diff = [r["diff_state_loss"] for r in regularized]
    reg_diff_errs = [r.get("diff_state_stderr", 0) for r in regularized]

    ax.axhline(baseline_diff, color="tab:gray", linestyle="--", linewidth=2,
               label=f"Baseline: {baseline_diff:.3f}")
    ax.errorbar(lambdas, reg_diff, yerr=reg_diff_errs, fmt="s-", color="tab:orange",
                markersize=8, linewidth=2, capsize=4, label="Regularized")
    ax.set_xscale("log")
    ax.set_xlabel("$\\lambda$ (contrastive weight)", fontsize=12)
    ax.set_ylabel("Different-State Loss (higher = better)", fontsize=12)
    ax.set_title("Different-State Discrimination", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_dir / f"direction_b_ib_vs_lambda.{args.format}",
                dpi=150, bbox_inches="tight")
    plt.close()

    # ---- Figure 2: Same-state vs different-state (pseudo-state breakdown) ----
    fig, ax = plt.subplots(figsize=(8, 6))

    categories = ["Same State\n(IB)", "Diff State\nSame Pseudo", "Diff State\nDiff Pseudo"]
    baseline_vals = [
        baseline["same_state_ib"],
        baseline.get("diff_state_same_pseudo_state_loss", 0),
        baseline.get("diff_state_diff_pseudo_state_loss", 0),
    ]

    x = np.arange(len(categories))
    width = 0.15
    ax.bar(x - width, baseline_vals, width, label="Baseline", color="tab:gray", alpha=0.8)

    for i, (lam, reg) in enumerate(zip(lambdas, regularized)):
        reg_vals = [
            reg["same_state_ib"],
            reg.get("diff_state_same_pseudo_state_loss", 0),
            reg.get("diff_state_diff_pseudo_state_loss", 0),
        ]
        ax.bar(x + width * i, reg_vals, width,
               label=f"$\\lambda$={lam}", alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_ylabel("IB Metric Value", fontsize=12)
    ax.set_title("IB Metrics: Baseline vs Regularized", fontsize=13)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    fig.savefig(output_dir / f"direction_b_ib_breakdown.{args.format}",
                dpi=150, bbox_inches="tight")
    plt.close()

    # Print summary
    logger.info("=" * 60)
    logger.info("Direction B: IB Comparison Summary")
    logger.info("=" * 60)
    logger.info(f"Baseline same-state IB: {baseline['same_state_ib']:.4f}")
    for lam, reg in zip(lambdas, regularized):
        improvement = reg["same_state_ib"] - baseline["same_state_ib"]
        logger.info(f"  lambda={lam}: same-state IB = {reg['same_state_ib']:.4f} "
                     f"(delta = {improvement:+.4f})")

    # Save summary
    summary = {
        "baseline": baseline,
        "regularized": {str(lam): reg for lam, reg in zip(lambdas, regularized)},
    }
    with open(output_dir / "direction_b_summary.json", "w") as f:
        json.dump(summary, f, indent=4)

    logger.info(f"Figures saved to {output_dir}")


if __name__ == "__main__":
    main()
