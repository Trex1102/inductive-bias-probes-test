"""Direction A: Compute and fit IB scaling laws from experiment results.

Fits power laws: IB(N) = alpha * N^{-beta} and L(N) = alpha * N^{-beta}
to study how inductive bias scales differently from loss.
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def power_law(x, alpha, beta):
    """Power law: y = alpha * x^{-beta}."""
    return alpha * np.power(x, -beta)


def log_linear_fit(x, y):
    """Fit log(y) = log(alpha) - beta * log(x) via linear regression.

    Returns (alpha, beta, r_squared).
    """
    log_x = np.log(x)
    log_y = np.log(y)

    # Linear regression: log_y = a + b * log_x
    A = np.vstack([log_x, np.ones(len(log_x))]).T
    result = np.linalg.lstsq(A, log_y, rcond=None)
    b, a = result[0]

    alpha = np.exp(a)
    beta = -b  # y = alpha * x^{-beta} means log_y = log_alpha - beta * log_x

    # R-squared
    y_pred = a + b * log_x
    ss_res = np.sum((log_y - y_pred) ** 2)
    ss_tot = np.sum((log_y - log_y.mean()) ** 2)
    r_squared = 1 - ss_res / (ss_tot + 1e-10)

    return alpha, beta, r_squared


def parse_args():
    parser = argparse.ArgumentParser(description="Compute IB scaling laws")
    parser.add_argument("--results_dir", type=str, required=True,
                        help="Directory containing scaling experiment results")
    parser.add_argument("--model_type", type=str, default="gpt")
    parser.add_argument("--output_file", type=str, default="scaling_laws.json")
    return parser.parse_args()


def main():
    args = parse_args()
    results_dir = Path(args.results_dir)

    # Load NTP loss results
    ntp_file = results_dir / f"scaling_ntp_results_{args.model_type}.json"
    if not ntp_file.exists():
        logger.error(f"NTP results not found: {ntp_file}")
        return

    with open(ntp_file) as f:
        ntp_results = json.load(f)

    # Load IB results for each scale
    ib_results = {}
    for label, info in ntp_results.items():
        ib_file = (
            results_dir / "white_noise" / args.model_type / "pt_next_token"
            / f"scaling_{label}" / "100_examples" / "100_iters" / "ib.json"
        )
        if ib_file.exists():
            with open(ib_file) as f:
                ib_results[label] = json.load(f)

    if not ib_results:
        logger.error("No IB results found. Run IB computation first.")
        return

    # Extract data for fitting
    param_counts = []
    losses = []
    ib_same_state = []
    ib_diff_state = []

    for label in sorted(ntp_results.keys()):
        if label not in ib_results:
            continue
        info = ntp_results[label]
        n_layer = info["n_layer"]
        n_embd = info["n_embd"]
        # Approximate parameter count (transformer): ~12 * n_layer * n_embd^2
        n_params = 12 * n_layer * n_embd ** 2
        param_counts.append(n_params)
        losses.append(info["val_loss"])
        ib_same_state.append(ib_results[label]["same_state_ib"])
        ib_diff_state.append(ib_results[label]["diff_state_loss"])

    param_counts = np.array(param_counts, dtype=np.float64)
    losses = np.array(losses, dtype=np.float64)
    ib_same_state = np.array(ib_same_state, dtype=np.float64)
    ib_diff_state = np.array(ib_diff_state, dtype=np.float64)

    logger.info(f"Fitting scaling laws with {len(param_counts)} data points")

    # Fit loss scaling: L(N) = alpha * N^{-beta}
    try:
        loss_alpha, loss_beta, loss_r2 = log_linear_fit(param_counts, losses)
        logger.info(f"Loss scaling: L(N) = {loss_alpha:.4f} * N^{{{-loss_beta:.4f}}}, "
                     f"R² = {loss_r2:.4f}")
    except Exception as e:
        logger.warning(f"Loss scaling fit failed: {e}")
        loss_alpha, loss_beta, loss_r2 = None, None, None

    # Fit IB scaling: IB(N) = alpha * N^{-beta}
    # Use 1 - same_state_ib as the "IB error" (want it to go down)
    ib_error = 1.0 - ib_same_state
    ib_error = np.clip(ib_error, 1e-6, None)  # avoid log(0)

    try:
        ib_alpha, ib_beta, ib_r2 = log_linear_fit(param_counts, ib_error)
        logger.info(f"IB scaling: IB_err(N) = {ib_alpha:.4f} * N^{{{-ib_beta:.4f}}}, "
                     f"R² = {ib_r2:.4f}")
    except Exception as e:
        logger.warning(f"IB scaling fit failed: {e}")
        ib_alpha, ib_beta, ib_r2 = None, None, None

    # Compute IB gap: divergence between loss and IB scaling
    if loss_beta is not None and ib_beta is not None:
        ib_gap = abs(loss_beta - ib_beta)
        logger.info(f"IB Gap (|beta_loss - beta_ib|): {ib_gap:.4f}")
        if ib_beta < loss_beta:
            logger.info("IB scales SLOWER than loss — IB gap grows with scale!")
        else:
            logger.info("IB scales FASTER than loss — scaling helps world models.")
    else:
        ib_gap = None

    # Save results
    output = {
        "loss_scaling": {
            "alpha": loss_alpha, "beta": loss_beta, "r_squared": loss_r2
        },
        "ib_scaling": {
            "alpha": ib_alpha, "beta": ib_beta, "r_squared": ib_r2
        },
        "ib_gap": ib_gap,
        "data_points": {
            label: {
                "n_params": int(pc),
                "loss": float(l),
                "ib_same_state": float(ib_s),
                "ib_diff_state": float(ib_d),
            }
            for label, pc, l, ib_s, ib_d in zip(
                sorted(ntp_results.keys()),
                param_counts, losses, ib_same_state, ib_diff_state,
            )
        },
    }

    output_path = results_dir / args.output_file
    with open(output_path, "w") as f:
        json.dump(output, f, indent=4)

    logger.info(f"Saved scaling laws to {output_path}")


if __name__ == "__main__":
    main()
