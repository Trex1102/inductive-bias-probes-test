"""Direction C: Compute IB metrics at each log-spaced checkpoint.

For each saved checkpoint, loads the model, fine-tunes on white noise datasets,
generates extrapolations, and computes IB metrics. The result is a time series
of IB metrics throughout training.
"""

import argparse
import json
import logging
import warnings
from pathlib import Path

import numpy as np
import torch

from inductivebiasprobes.paths import (
    GRIDWORLD_CKPT_DIR,
    GRIDWORLD_DATA_DIR,
    GRIDWORLD_EXT_DIR,
    HEAT_EQ_CKPT_DIR,
    HEAT_EQ_DATA_DIR,
    HEAT_EQ_EXT_DIR,
)
from inductivebiasprobes.src.contrastive_regularizer import get_log_spaced_steps
from inductivebiasprobes.src.model import Model, ModelConfig
from inductivebiasprobes.src.train_utils import (
    generate_and_save_extrapolations,
    get_batch,
    get_lr,
    setup_training_environment,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


DOMAIN_CONFIGS = {
    "gridworld": {
        "ckpt_dir": GRIDWORLD_CKPT_DIR,
        "data_dir": GRIDWORLD_DATA_DIR,
        "ext_dir": GRIDWORLD_EXT_DIR,
    },
    "heat_equation": {
        "ckpt_dir": HEAT_EQ_CKPT_DIR,
        "data_dir": HEAT_EQ_DATA_DIR,
        "ext_dir": HEAT_EQ_EXT_DIR,
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Direction C: Compute IB dynamics across checkpoints"
    )
    parser.add_argument("--domain", type=str, required=True,
                        choices=["gridworld", "heat_equation"])
    parser.add_argument("--model_type", type=str, default="gpt",
                        choices=["gpt", "mamba", "mamba2", "rnn", "lstm"])
    parser.add_argument("--num_states", type=int, default=5)
    parser.add_argument("--max_iters", type=int, default=60000,
                        help="Max iters of the original training run")
    parser.add_argument("--wn_max_iters", type=int, default=100,
                        help="Max iters for white noise fine-tuning")
    parser.add_argument("--num_white_noise_datasets", type=int, default=20,
                        help="Number of WN datasets per checkpoint")
    parser.add_argument("--white_noise_dataset_size", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--checkpoint_source", type=str, default="dynamics",
                        help="Subdirectory name under model_type for checkpoints")
    return parser.parse_args()


def load_model_from_checkpoint(ckpt_path, device):
    """Load a model from a specific checkpoint file."""
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model_args = checkpoint["model_args"]

    # Build ModelConfig from saved args
    config_fields = {f.name for f in ModelConfig.__dataclass_fields__.values()}
    model_config_args = {k: v for k, v in model_args.items() if k in config_fields}
    model_config = ModelConfig(**model_config_args)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*torch.cuda.amp.*")
        model = Model(model_config)

    state_dict = checkpoint["model"]
    unwanted_prefix = "_orig_mod."
    for k in list(state_dict.keys()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

    model.load_state_dict(state_dict)
    model = model.to(device)
    return model, model_args


def finetune_on_white_noise(model, wn_config, wn_max_iters, device):
    """Fine-tune model on a single white noise dataset (lightweight)."""
    import torch._dynamo as dynamo
    dynamo.config.disable = True

    model.train()
    # Update model config for white noise output dimensions
    model.config.output_vocab_size = wn_config.get("output_vocab_size", 2)
    model.config.output_dim = wn_config.get("output_dim", 1)
    model.config.mask_id = wn_config.get("mask_id", -1)
    # Reinitialize output head with new dimensions
    model.reset_output_head()
    model = model.to(device)

    optimizer = model.configure_optimizers(
        weight_decay=0.1, learning_rate=6e-4,
        betas=(0.9, 0.95), device_type="cuda" if "cuda" in device else "cpu",
    )

    for step in range(wn_max_iters):
        X, Y = get_batch("train", wn_config)
        _, loss = model(X, Y)
        loss = loss.mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    return model


def compute_ib_at_checkpoint(
    ckpt_path, args, domain_cfg, wn_data_dir, ext_base_dir, step
):
    """Compute IB metrics for a single checkpoint."""
    logger.info(f"  Loading checkpoint: {ckpt_path.name}")

    model, model_args = load_model_from_checkpoint(ckpt_path, args.device)

    # Get the NTP validation loss from this checkpoint
    checkpoint = torch.load(ckpt_path, map_location=args.device, weights_only=False)
    ntp_val_loss = checkpoint.get("best_val_loss", None)

    all_extrapolations = []

    for wn_idx in range(args.num_white_noise_datasets):
        # Create a fresh copy of the model for each WN dataset
        model_copy, _ = load_model_from_checkpoint(ckpt_path, args.device)

        # Setup WN config
        wn_config = dict(model_args)
        wn_size = args.white_noise_dataset_size
        wn_dir = wn_data_dir / f"{wn_size}-examples"
        wn_config["train_file"] = wn_dir / f"white_noise_obs_train_{wn_idx}.npy"
        wn_config["val_file"] = wn_dir / f"white_noise_obs_val_{wn_idx}.npy"
        wn_config["train_target_file"] = wn_dir / f"white_noise_output_train_{wn_idx}.npy"
        wn_config["val_target_file"] = wn_dir / f"white_noise_states_val_{wn_idx}.npy"
        wn_config["train_indices_file"] = wn_dir / f"white_noise_indices_train_{wn_idx}.npy"
        wn_config["val_indices_file"] = wn_dir / f"white_noise_indices_val_{wn_idx}.npy"
        wn_config["predict_type"] = "white_noise"
        wn_config["output_vocab_size"] = 2
        wn_config["output_dim"] = 1
        wn_config["num_data_points"] = wn_size
        wn_config["batch_size"] = min(128, wn_size)
        wn_config["use_float_x"] = False
        wn_config["use_float_y"] = False
        wn_config["mask_id"] = -1
        wn_config["device"] = args.device
        wn_config["device_type"] = "cuda" if "cuda" in args.device else "cpu"

        # Fine-tune on white noise
        model_copy = finetune_on_white_noise(
            model_copy, wn_config, args.wn_max_iters, args.device
        )

        # Generate extrapolations
        ext_dir = ext_base_dir / "white_noise"
        ext_idx_dir = (
            ext_dir / args.model_type / "dynamics"
            / f"step_{step}" / f"idx_{wn_idx}"
        )
        generate_and_save_extrapolations(model_copy, wn_config, ext_dir, ext_idx_dir)

        # Load extrapolation for IB computation
        ext_probs = np.load(ext_idx_dir / "extrapolations.npy")
        ext = np.argmax(ext_probs, axis=-1).ravel()
        all_extrapolations.append(ext)

        del model_copy
        torch.cuda.empty_cache()

    # Compute IB metrics from extrapolations
    states = np.load(ext_dir / "states.npy").astype(np.int32).ravel()
    all_ext = np.stack(all_extrapolations, axis=1)

    n = min(2000, len(states))
    rng = np.random.RandomState(0)
    idx = rng.choice(len(states), size=n, replace=False)
    states_sub = states[idx]
    ext_sub = all_ext[idx]

    # Same-state metric
    triu = np.triu(np.ones((n, n)), k=1).astype(bool)
    same_mask = (states_sub[:, None] == states_sub[None, :]) & triu
    diff_mask = (states_sub[:, None] != states_sub[None, :]) & triu

    x1 = ext_sub[:, None, :]
    x2 = ext_sub[None, :, :]
    pred = np.where(x1 == 1, 1, 0)
    losses = (pred != x2).astype(np.float64)

    same_vals = (2 * (0.5 - losses[same_mask])).mean(axis=0)
    same_state_ib = float(same_vals.mean())
    same_state_se = float(same_vals.std(ddof=1) / np.sqrt(len(same_vals)))

    diff_vals = (2 * losses[diff_mask]).mean(axis=0)
    diff_state_loss = float(diff_vals.mean())
    diff_state_se = float(diff_vals.std(ddof=1) / np.sqrt(len(diff_vals)))

    del model
    torch.cuda.empty_cache()

    return {
        "step": step,
        "ntp_val_loss": float(ntp_val_loss) if ntp_val_loss is not None else None,
        "same_state_ib": same_state_ib,
        "same_state_stderr": same_state_se,
        "diff_state_loss": diff_state_loss,
        "diff_state_stderr": diff_state_se,
    }


def main():
    args = parse_args()
    domain_cfg = DOMAIN_CONFIGS[args.domain]

    if args.domain == "gridworld":
        ckpt_base = domain_cfg["ckpt_dir"] / f"{args.num_states}-states"
        data_dir = domain_cfg["data_dir"] / f"{args.num_states}-states"
        ext_base = domain_cfg["ext_dir"] / f"{args.num_states}-states"
    else:
        ckpt_base = domain_cfg["ckpt_dir"]
        data_dir = domain_cfg["data_dir"]
        ext_base = domain_cfg["ext_dir"]

    ckpt_dir = ckpt_base / args.model_type / args.checkpoint_source / "next_token"
    wn_data_dir = data_dir / "white_noise"

    # Find available checkpoints
    checkpoint_steps = get_log_spaced_steps(args.max_iters)
    available_ckpts = []
    for step in checkpoint_steps:
        ckpt_path = ckpt_dir / f"ckpt_step_{step}.pt"
        if ckpt_path.exists():
            available_ckpts.append((step, ckpt_path))

    if not available_ckpts:
        logger.error(f"No checkpoints found in {ckpt_dir}")
        logger.error(f"Expected files like: ckpt_step_100.pt, ckpt_step_1000.pt, ...")
        return

    logger.info(f"Found {len(available_ckpts)} checkpoints")

    # Compute IB at each checkpoint
    dynamics = []
    for step, ckpt_path in available_ckpts:
        logger.info(f"Processing step {step}...")
        result = compute_ib_at_checkpoint(
            ckpt_path, args, domain_cfg, wn_data_dir, ext_base, step
        )
        dynamics.append(result)
        logger.info(f"  Step {step}: same_state_ib={result['same_state_ib']:.4f}, "
                     f"ntp_loss={result['ntp_val_loss']}")

    # Save dynamics
    output_dir = ext_base / "ib_dynamics" / args.model_type / args.checkpoint_source
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "ib_dynamics.json", "w") as f:
        json.dump(dynamics, f, indent=4)

    logger.info(f"Saved IB dynamics to {output_dir / 'ib_dynamics.json'}")

    # Quick summary
    steps = [d["step"] for d in dynamics]
    ibs = [d["same_state_ib"] for d in dynamics]
    logger.info(f"IB trajectory: {list(zip(steps, [f'{ib:.3f}' for ib in ibs]))}")


if __name__ == "__main__":
    main()
