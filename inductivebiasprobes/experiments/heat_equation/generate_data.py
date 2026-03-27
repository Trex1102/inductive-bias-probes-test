"""Generate 1D heat equation trajectory data for pretraining and evaluation."""

import argparse
import logging

import numpy as np
import yaml

from inductivebiasprobes.paths import HEAT_EQ_CONFIG_DIR, HEAT_EQ_DATA_DIR
from inductivebiasprobes.src.heat_equation import (
    discretize_temperatures,
    generate_heat_trajectories,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Default diffusivity values defining the state space (5 states)
ALPHA_VALUES = [0.002, 0.005, 0.01, 0.02, 0.05]


def parse_args():
    parser = argparse.ArgumentParser(description="Generate heat equation data")
    parser.add_argument("--num_train_sequences", type=int, default=100_000)
    parser.add_argument("--num_val_sequences", type=int, default=1_000)
    parser.add_argument("--N_x", type=int, default=21, help="Spatial grid points")
    parser.add_argument("--N_t", type=int, default=101, help="Time steps")
    parser.add_argument("--dt", type=float, default=0.01, help="Time step size")
    parser.add_argument("--num_modes", type=int, default=5, help="Fourier modes")
    parser.add_argument("--num_bins", type=int, default=50, help="Temperature bins")
    parser.add_argument("--num_obs", type=int, default=5, help="Observation points")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    rng = np.random.RandomState(args.seed)

    obs_indices = np.linspace(1, args.N_x - 2, args.num_obs).astype(int)
    num_states = len(ALPHA_VALUES)
    logger.info(f"Generating heat equation data: {num_states} alpha values, "
                f"{args.num_obs} obs points, {args.N_t} time steps")

    # Generate training data
    logger.info("Generating training trajectories...")
    obs_train, states_train, _ = generate_heat_trajectories(
        num_sequences=args.num_train_sequences,
        alpha_values=ALPHA_VALUES,
        N_x=args.N_x,
        obs_indices=obs_indices,
        N_t=args.N_t,
        dt=args.dt,
        num_modes=args.num_modes,
        rng=rng,
    )

    # Generate validation data
    logger.info("Generating validation trajectories...")
    obs_val, states_val, _ = generate_heat_trajectories(
        num_sequences=args.num_val_sequences,
        alpha_values=ALPHA_VALUES,
        N_x=args.N_x,
        obs_indices=obs_indices,
        N_t=args.N_t,
        dt=args.dt,
        num_modes=args.num_modes,
        rng=rng,
    )

    # Discretize temperatures using training data bin edges
    logger.info("Discretizing temperatures...")
    obs_train_disc, bin_edges = discretize_temperatures(
        obs_train, num_bins=args.num_bins
    )
    obs_val_disc, _ = discretize_temperatures(
        obs_val, num_bins=args.num_bins, bin_edges=bin_edges
    )

    # Save data
    data_dir = HEAT_EQ_DATA_DIR
    data_dir.mkdir(parents=True, exist_ok=True)

    np.save(data_dir / "obs_train.npy", obs_train_disc.astype(np.uint8))
    np.save(data_dir / "obs_val.npy", obs_val_disc.astype(np.uint8))
    np.save(data_dir / "state_train.npy", states_train.astype(np.uint8))
    np.save(data_dir / "state_val.npy", states_val.astype(np.uint8))
    np.save(data_dir / "bin_edges.npy", bin_edges)
    np.save(data_dir / "obs_train_continuous.npy", obs_train.astype(np.float32))
    np.save(data_dir / "obs_val_continuous.npy", obs_val.astype(np.float32))

    logger.info(f"Saved data to {data_dir}")
    logger.info(f"  obs_train: {obs_train_disc.shape}, "
                f"obs_val: {obs_val_disc.shape}")
    logger.info(f"  state_train: {states_train.shape}, "
                f"states range: {states_train.min()}-{states_train.max()}")

    # Save configs
    config_dir = HEAT_EQ_CONFIG_DIR
    config_dir.mkdir(parents=True, exist_ok=True)

    token_vocab_size = args.num_bins
    block_size = args.N_t - 1  # 100

    ntp_config = {
        "input_dim": args.num_obs,
        "output_dim": args.num_obs,
        "input_vocab_size": token_vocab_size,
        "output_vocab_size": token_vocab_size,
        "mask_id": -1,
        "block_size": block_size,
        "num_data_points": args.num_train_sequences,
        "predict_type": "next_token",
    }

    state_config = {
        "input_dim": args.num_obs,
        "output_dim": 1,
        "input_vocab_size": token_vocab_size,
        "output_vocab_size": num_states,
        "mask_id": -1,
        "block_size": block_size,
        "num_data_points": args.num_train_sequences,
        "predict_type": "state",
    }

    with open(config_dir / "ntp_config.yaml", "w") as f:
        yaml.dump(ntp_config, f)
    with open(config_dir / "state_config.yaml", "w") as f:
        yaml.dump(state_config, f)

    logger.info(f"Saved configs to {config_dir}")
    logger.info(f"  vocab_size: {token_vocab_size}, block_size: {block_size}, "
                f"input_dim: {args.num_obs}")


if __name__ == "__main__":
    main()
