"""Compute inductive bias metrics for heat equation models."""

import argparse
import json
import logging

import numpy as np

from inductivebiasprobes.paths import HEAT_EQ_EXT_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

NUM_STATES = 5


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default="gpt",
                        choices=["gpt", "mamba", "mamba2", "rnn", "lstm"])
    parser.add_argument("--pretrained", type=str, default="scratch",
                        choices=["scratch", "next_token"])
    parser.add_argument("--white_noise_dataset_size", type=int, default=100)
    parser.add_argument("--num_white_noise_datasets", type=int, default=100)
    parser.add_argument("--max_iters", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_examples", type=int, default=2000)
    return parser.parse_args()


def build_pseudo_states(state):
    """For heat equation, pseudo-states are based on state boundaries.

    States 0 and num_states-1 have different transition probabilities
    (like gridworld boundaries), while interior states are equivalent.
    """
    if state == 0:
        return 0  # boundary
    elif state == NUM_STATES - 1:
        return 2  # boundary
    else:
        return 1  # interior


def _compute_discrete_inductive_bias(ext_train, ext_test, states, pseudo_states):
    """Compute discrete inductive bias (same logic as gridworld)."""
    n = len(states)
    triu_mask = np.triu(np.ones((n, n)), k=1).astype(bool)

    state_matrix = states[:, np.newaxis] == states[np.newaxis, :]
    pseudo_state_matrix = pseudo_states[:, np.newaxis] == pseudo_states[np.newaxis, :]
    same_state_mask = state_matrix & triu_mask
    diff_state_mask = (~state_matrix) & triu_mask
    diff_state_same_pseudo_mask = diff_state_mask & pseudo_state_matrix
    diff_state_diff_pseudo_mask = diff_state_mask & (~pseudo_state_matrix)

    x1_test = ext_test[:, np.newaxis, :]
    x2_test = ext_test[np.newaxis, :, :]
    pred = np.where(x1_test == 1, 1, 0)
    losses = (pred != x2_test).astype(np.float64)

    def compute_metric(mask):
        vals = (2 * (0.5 - losses[mask])).mean(axis=0)
        return vals.mean(), vals.std(ddof=1) / np.sqrt(len(vals))

    def compute_diff_metric(mask):
        vals = (2 * losses[mask]).mean(axis=0)
        return vals.mean(), vals.std(ddof=1) / np.sqrt(len(vals))

    same_val, same_se = compute_metric(same_state_mask)
    diff_val, diff_se = compute_diff_metric(diff_state_mask)

    if np.sum(diff_state_same_pseudo_mask) > 0:
        diff_same_ps_val, diff_same_ps_se = compute_diff_metric(diff_state_same_pseudo_mask)
    else:
        diff_same_ps_val, diff_same_ps_se = 0.0, 0.0

    if np.sum(diff_state_diff_pseudo_mask) > 0:
        diff_diff_ps_val, diff_diff_ps_se = compute_diff_metric(diff_state_diff_pseudo_mask)
    else:
        diff_diff_ps_val, diff_diff_ps_se = 0.0, 0.0

    return (same_val, same_se, diff_val, diff_se,
            diff_same_ps_val, diff_same_ps_se, diff_diff_ps_val, diff_diff_ps_se)


def main():
    args = parse_args()
    rng = np.random.RandomState(args.seed)

    ext_dir = HEAT_EQ_EXT_DIR / "white_noise"
    states = np.load(ext_dir / "states.npy").astype(np.int32).ravel()
    pseudo_states = np.array([build_pseudo_states(s) for s in states])

    ext_curr_dir = (
        ext_dir
        / args.model_type
        / f"pt_{args.pretrained}"
        / f"{args.white_noise_dataset_size}_examples"
        / f"{args.max_iters}_iters"
    )

    test_seeds = rng.choice(
        range(args.num_white_noise_datasets),
        args.num_white_noise_datasets,
        replace=False,
    )
    train_seeds = np.setdiff1d(range(args.num_white_noise_datasets), test_seeds)

    all_ext = []
    for seed in range(args.num_white_noise_datasets):
        ext_probs = np.load(ext_curr_dir / f"idx_{seed}" / "extrapolations.npy")
        ext = np.argmax(ext_probs, axis=-1).ravel()
        all_ext.append(ext)

    all_ext = np.stack(all_ext, axis=1)
    ext_train = all_ext[:, train_seeds] if len(train_seeds) > 0 else all_ext
    ext_test = all_ext[:, test_seeds]

    # Subsample for computational efficiency
    example_indices = rng.choice(len(states), size=min(args.num_examples, len(states)),
                                  replace=False)
    ext_train = ext_train[example_indices]
    ext_test = ext_test[example_indices]
    states_sub = states[example_indices]
    pseudo_sub = pseudo_states[example_indices]

    results = _compute_discrete_inductive_bias(ext_train, ext_test, states_sub, pseudo_sub)
    (same_val, same_se, diff_val, diff_se,
     diff_same_ps_val, diff_same_ps_se, diff_diff_ps_val, diff_diff_ps_se) = results

    logger.info(f"Same-state IB: {same_val:.3f} ± {same_se:.3f}")
    logger.info(f"Diff-state loss: {diff_val:.3f} ± {diff_se:.3f}")
    logger.info(f"Diff-state same-pseudo: {diff_same_ps_val:.3f} ± {diff_same_ps_se:.3f}")
    logger.info(f"Diff-state diff-pseudo: {diff_diff_ps_val:.3f} ± {diff_diff_ps_se:.3f}")

    ib_results = {
        "same_state_ib": float(same_val),
        "same_state_stderr": float(same_se),
        "diff_state_loss": float(diff_val),
        "diff_state_stderr": float(diff_se),
        "diff_state_same_pseudo_state_loss": float(diff_same_ps_val),
        "diff_state_same_pseudo_state_stderr": float(diff_same_ps_se),
        "diff_state_diff_pseudo_state_loss": float(diff_diff_ps_val),
        "diff_state_diff_pseudo_state_stderr": float(diff_diff_ps_se),
    }
    with open(ext_curr_dir / "ib.json", "w") as f:
        json.dump(ib_results, f, indent=4)

    logger.info(f"Saved results to {ext_curr_dir / 'ib.json'}")


if __name__ == "__main__":
    main()
