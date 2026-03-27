"""Generate white noise datasets for the heat equation IB probe."""

import argparse
import logging

import numpy as np
import tqdm
import yaml

from inductivebiasprobes.paths import HEAT_EQ_CONFIG_DIR, HEAT_EQ_DATA_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

NUM_STATES = 5  # Number of alpha values (state space size)


def generate_state_dependent_noise(
    num_states, states_train, states_val, random_seed
):
    """Generate noise that is a deterministic function of the state (alpha index).

    Args:
        num_states: Number of possible states.
        states_train: shape (N_train, seq_len, 1) containing state indices.
        states_val: shape (N_val, seq_len, 1) containing state indices.
        random_seed: Random seed for reproducibility.

    Returns:
        noise_train, indices_train, noise_val, indices_val
    """
    rng = np.random.RandomState(random_seed)
    train_size, seq_len, _ = states_train.shape
    val_size = states_val.shape[0]

    # Random binary label per state
    noise_lookup = {state: rng.choice([0, 1]) for state in range(num_states)}

    noise_train = np.zeros((train_size, seq_len, 1))
    noise_val = np.zeros((val_size, seq_len, 1))

    for state, label in noise_lookup.items():
        noise_train[states_train == state] = label
        noise_val[states_val == state] = label

    # Random sequence indices for masking
    indices_train = rng.randint(0, seq_len - 1, size=train_size)
    indices_val = rng.randint(0, seq_len - 1, size=val_size)

    return noise_train, indices_train, noise_val, indices_val


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate heat equation white noise data"
    )
    parser.add_argument("--num_white_noise_datasets", type=int, default=100)
    parser.add_argument("--white_noise_dataset_size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    rng = np.random.RandomState(args.seed)

    data_dir = HEAT_EQ_DATA_DIR
    config_dir = HEAT_EQ_CONFIG_DIR

    # Load pre-generated data
    train_obs = np.load(data_dir / "obs_train.npy")
    val_obs = np.load(data_dir / "obs_val.npy")
    train_states = np.load(data_dir / "state_train.npy")
    val_states = np.load(data_dir / "state_val.npy")

    logger.info(f"Loaded data: train={train_obs.shape}, val={val_obs.shape}")

    # Create white noise directory
    noise_dir = data_dir / "white_noise" / f"{args.white_noise_dataset_size}-examples"
    noise_dir.mkdir(parents=True, exist_ok=True)

    # Sample validation indices once (reuse for all datasets)
    val_dataset_indices = rng.choice(
        len(val_states), size=args.white_noise_dataset_size, replace=False
    )

    for idx in tqdm.trange(args.num_white_noise_datasets, desc="Generating WN"):
        # Sample training indices
        train_dataset_indices = rng.choice(
            len(train_states), size=args.white_noise_dataset_size, replace=False
        )

        train_dataset_obs = train_obs[train_dataset_indices]
        val_dataset_obs = val_obs[val_dataset_indices]
        train_dataset_states = train_states[train_dataset_indices]
        val_dataset_states = val_states[val_dataset_indices]

        # Generate state-dependent noise
        train_noise, train_indices, val_noise, val_indices = (
            generate_state_dependent_noise(
                NUM_STATES,
                train_dataset_states,
                val_dataset_states,
                args.seed + idx * 100,
            )
        )

        # Save all arrays
        np.save(noise_dir / f"white_noise_output_train_{idx}.npy", train_noise)
        np.save(noise_dir / f"white_noise_output_val_{idx}.npy", val_noise)
        np.save(noise_dir / f"white_noise_obs_train_{idx}.npy", train_dataset_obs)
        np.save(noise_dir / f"white_noise_obs_val_{idx}.npy", val_dataset_obs)
        np.save(noise_dir / f"white_noise_states_train_{idx}.npy", train_dataset_states)
        np.save(noise_dir / f"white_noise_states_val_{idx}.npy", val_dataset_states)
        np.save(noise_dir / f"white_noise_indices_train_{idx}.npy", train_indices)
        np.save(noise_dir / f"white_noise_indices_val_{idx}.npy", val_indices)

    # Save white noise config
    config = {
        "input_dim": train_obs.shape[-1],
        "output_dim": 1,
        "input_vocab_size": int(np.max(train_obs)) + 1,
        "block_size": train_obs.shape[1] - 1,
        "mask_id": -1,
        "predict_type": "white_noise",
        "output_vocab_size": 2,
        "num_data_points": args.white_noise_dataset_size,
    }
    with open(config_dir / "white_noise_config.yaml", "w") as f:
        yaml.dump(config, f)

    logger.info(f"Saved {args.num_white_noise_datasets} white noise datasets to "
                f"{noise_dir}")


if __name__ == "__main__":
    main()
