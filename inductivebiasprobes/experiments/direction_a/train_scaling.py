"""Direction A: Train models at various scales to measure IB scaling laws.

Trains models with different numbers of layers and embedding dimensions,
then runs IB probes on each to measure how IB metrics scale with model size.
"""

import argparse
import itertools
import json
import logging
import os

import numpy as np
import yaml

from inductivebiasprobes.paths import (
    GRIDWORLD_CKPT_DIR,
    GRIDWORLD_CONFIG_DIR,
    GRIDWORLD_DATA_DIR,
    GRIDWORLD_EXT_DIR,
    HEAT_EQ_CKPT_DIR,
    HEAT_EQ_CONFIG_DIR,
    HEAT_EQ_DATA_DIR,
    HEAT_EQ_EXT_DIR,
)
from inductivebiasprobes.src.train_utils import (
    add_common_args,
    generate_and_save_extrapolations,
    init_model,
    setup_training_environment,
    train,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model size grid for scaling experiments
LAYER_SIZES = [2, 4, 6, 8, 12]
EMBED_SIZES = [64, 128, 256, 512]

# Dataset size grid
DATA_SIZES = [1000, 10000, 50000, 100000]

# Domain configurations
DOMAIN_CONFIGS = {
    "gridworld": {
        "ckpt_dir": GRIDWORLD_CKPT_DIR,
        "config_dir": GRIDWORLD_CONFIG_DIR,
        "data_dir": GRIDWORLD_DATA_DIR,
        "ext_dir": GRIDWORLD_EXT_DIR,
        "num_states": 5,
        "ntp_config": "ntp_config",
        "wn_config": "white_noise_config",
    },
    "heat_equation": {
        "ckpt_dir": HEAT_EQ_CKPT_DIR,
        "config_dir": HEAT_EQ_CONFIG_DIR,
        "data_dir": HEAT_EQ_DATA_DIR,
        "ext_dir": HEAT_EQ_EXT_DIR,
        "ntp_config": "ntp_config",
        "wn_config": "white_noise_config",
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description="Direction A: IB Scaling Laws")
    parser.add_argument("--domain", type=str, required=True,
                        choices=["gridworld", "heat_equation"])
    parser.add_argument("--model_type", type=str, default="gpt",
                        choices=["gpt", "mamba", "rnn", "lstm"])
    parser.add_argument("--mode", type=str, required=True,
                        choices=["train_ntp", "train_wn", "compute_ib"],
                        help="train_ntp: pretrain at scale, "
                             "train_wn: white noise fine-tune, "
                             "compute_ib: compute IB metrics")
    parser.add_argument("--n_layer", type=int, default=None,
                        help="Specific layer count (if not running full grid)")
    parser.add_argument("--n_embd", type=int, default=None,
                        help="Specific embedding dim (if not running full grid)")
    parser.add_argument("--num_data_points", type=int, default=None,
                        help="Override dataset size for data scaling experiments")
    parser.add_argument("--max_iters", type=int, default=60000)
    parser.add_argument("--wn_max_iters", type=int, default=100,
                        help="Max iters for white noise fine-tuning")
    parser.add_argument("--num_white_noise_datasets", type=int, default=100)
    parser.add_argument("--white_noise_dataset_size", type=int, default=100)
    parser.add_argument("--num_states", type=int, default=5,
                        help="For gridworld: number of states")
    parser.add_argument("--no_wandb", action="store_true", default=True)
    parser.add_argument("--no_compile", action="store_true", default=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="float32")
    return parser.parse_args()


def get_model_configs(args):
    """Get list of (n_layer, n_embd) configs to run."""
    if args.n_layer is not None and args.n_embd is not None:
        return [(args.n_layer, args.n_embd)]
    elif args.n_layer is not None:
        return [(args.n_layer, e) for e in EMBED_SIZES]
    elif args.n_embd is not None:
        return [(l, args.n_embd) for l in LAYER_SIZES]
    else:
        return list(itertools.product(LAYER_SIZES, EMBED_SIZES))


def get_scale_label(n_layer, n_embd):
    return f"L{n_layer}_D{n_embd}"


def load_base_config(args, domain_cfg):
    """Load and return the base NTP config for a domain."""
    if args.domain == "gridworld":
        config_path = (
            domain_cfg["config_dir"] / f"{args.num_states}-states" / "ntp_config.yaml"
        )
    else:
        config_path = domain_cfg["config_dir"] / "ntp_config.yaml"

    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def load_wn_config(args, domain_cfg):
    """Load the white noise config for a domain."""
    if args.domain == "gridworld":
        config_path = (
            domain_cfg["config_dir"]
            / f"{args.num_states}-states"
            / "white_noise_config.yaml"
        )
    else:
        config_path = domain_cfg["config_dir"] / "white_noise_config.yaml"

    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def train_ntp_at_scale(args, n_layer, n_embd, domain_cfg):
    """Train an NTP model at a specific scale."""
    label = get_scale_label(n_layer, n_embd)
    logger.info(f"Training NTP: {label} on {args.domain}")

    config = load_base_config(args, domain_cfg)
    config.update({
        "model_type": args.model_type,
        "n_layer": n_layer,
        "n_embd": n_embd,
        "n_head": max(1, n_embd // 64) if args.model_type == "gpt" else None,
        "dropout": 0.0 if args.model_type == "gpt" else None,
        "bias": True,
        "max_iters": args.max_iters,
        "no_wandb": args.no_wandb,
        "no_compile": args.no_compile,
        "device": args.device,
        "dtype": args.dtype,
        "pretrained": "scratch",
        "eval_interval": 500,
        "eval_iters": 10,
        "batch_size": 128,
        "learning_rate": 6e-4,
        "weight_decay": 0.1,
        "beta1": 0.9,
        "beta2": 0.95,
        "grad_clip": 1.0,
        "decay_lr": False,
        "warmup_iters": 2000,
        "lr_decay_iters": args.max_iters,
        "min_lr": 6e-5,
        "log_interval": 100,
        "always_save_checkpoint": False,
        "backend": "nccl",
        "save_loss": False,
        "resume_from_last_ckpt": False,
        "plot_trajectory": False,
        "gradient_accumulation_steps": 1,
    })

    if args.num_data_points is not None:
        config["num_data_points"] = args.num_data_points

    # Set Mamba/RNN/LSTM-specific configs
    if args.model_type in ("mamba", "mamba2", "rnn", "lstm"):
        config.update({
            "dt_rank": "auto", "d_state": 16, "expand_factor": 2,
            "d_conv": 4, "dt_min": 0.001, "dt_max": 0.1,
            "dt_init": "random", "dt_scale": 1.0, "dt_init_floor": 1e-4,
            "rms_norm_eps": 1e-5, "conv_bias": True,
            "inner_layernorms": True, "pscan": True, "use_cuda": True,
            "n_head": None, "dropout": None,
        })
    else:
        for k in ["dt_rank", "d_state", "expand_factor", "d_conv", "dt_min",
                   "dt_max", "dt_init", "dt_scale", "dt_init_floor",
                   "rms_norm_eps", "conv_bias", "inner_layernorms", "pscan", "use_cuda"]:
            config[k] = None

    # Setup data paths
    if args.domain == "gridworld":
        data_dir = domain_cfg["data_dir"] / f"{args.num_states}-states"
    else:
        data_dir = domain_cfg["data_dir"]

    config["train_file"] = data_dir / "obs_train.npy"
    config["val_file"] = data_dir / "obs_val.npy"
    config["use_float_x"] = False
    config["use_float_y"] = config.get("output_vocab_size") is None

    # Checkpoint dirs
    scaling_dir = "scaling_" + label
    ckpt_dir = domain_cfg["ckpt_dir"]
    if args.domain == "gridworld":
        ckpt_dir = ckpt_dir / f"{args.num_states}-states"
    save_ckpt_dir = ckpt_dir / args.model_type / scaling_dir / "next_token"

    ddp, master_process, ptdtype, config = setup_training_environment(
        config, save_ckpt_dir, True
    )

    model, config, iter_num, current_epoch, best_val_loss, optimizer, scaler = (
        init_model(config=config, ckpt_dir=save_ckpt_dir, ddp=ddp)
    )

    train(
        model=model, optimizer=optimizer, scaler=scaler, config=config,
        ddp=ddp, master_process=master_process, ptdtype=ptdtype,
        iter_num=iter_num, current_epoch=current_epoch,
        best_val_loss=best_val_loss, ckpt_dir=save_ckpt_dir,
        save_checkpoints=True,
    )

    logger.info(f"Done training {label}. Best val loss: {best_val_loss:.6f}")
    return best_val_loss


def train_wn_at_scale(args, n_layer, n_embd, domain_cfg):
    """Train white noise models using a pretrained checkpoint at specific scale."""
    label = get_scale_label(n_layer, n_embd)
    logger.info(f"Training WN for scale {label}")

    wn_config = load_wn_config(args, domain_cfg)
    wn_config.update({
        "model_type": args.model_type,
        "n_layer": n_layer,
        "n_embd": n_embd,
        "n_head": max(1, n_embd // 64) if args.model_type == "gpt" else None,
        "dropout": 0.0 if args.model_type == "gpt" else None,
        "bias": True,
        "max_iters": args.wn_max_iters,
        "no_wandb": True,
        "no_compile": True,
        "device": args.device,
        "dtype": args.dtype,
        "pretrained": "next_token",
        "eval_interval": max(1, args.wn_max_iters // 5),
        "eval_iters": 5,
        "batch_size": min(128, wn_config.get("num_data_points", 100)),
        "learning_rate": 6e-4,
        "weight_decay": 0.1,
        "beta1": 0.9,
        "beta2": 0.95,
        "grad_clip": 1.0,
        "decay_lr": False,
        "warmup_iters": 0,
        "lr_decay_iters": args.wn_max_iters,
        "min_lr": 6e-5,
        "log_interval": 100,
        "always_save_checkpoint": False,
        "backend": "nccl",
        "save_loss": False,
        "resume_from_last_ckpt": False,
        "plot_trajectory": False,
        "gradient_accumulation_steps": 1,
        "num_white_noise_datasets": args.num_white_noise_datasets,
        "white_noise_dataset_size": args.white_noise_dataset_size,
    })

    if args.model_type in ("mamba", "mamba2", "rnn", "lstm"):
        wn_config.update({
            "dt_rank": "auto", "d_state": 16, "expand_factor": 2,
            "d_conv": 4, "dt_min": 0.001, "dt_max": 0.1,
            "dt_init": "random", "dt_scale": 1.0, "dt_init_floor": 1e-4,
            "rms_norm_eps": 1e-5, "conv_bias": True,
            "inner_layernorms": True, "pscan": True, "use_cuda": True,
            "n_head": None, "dropout": None,
        })
    else:
        for k in ["dt_rank", "d_state", "expand_factor", "d_conv", "dt_min",
                   "dt_max", "dt_init", "dt_scale", "dt_init_floor",
                   "rms_norm_eps", "conv_bias", "inner_layernorms", "pscan", "use_cuda"]:
            wn_config[k] = None

    scaling_dir = "scaling_" + label
    ckpt_dir = domain_cfg["ckpt_dir"]
    if args.domain == "gridworld":
        ckpt_dir = ckpt_dir / f"{args.num_states}-states"
        data_dir = domain_cfg["data_dir"] / f"{args.num_states}-states"
    else:
        data_dir = domain_cfg["data_dir"]

    pretrained_ckpt_dir = ckpt_dir / args.model_type / scaling_dir / "next_token"

    for dataset_idx in range(args.num_white_noise_datasets):
        logger.info(f"  WN dataset {dataset_idx}/{args.num_white_noise_datasets}")

        config = dict(wn_config)  # copy
        wn_dir = (
            data_dir / "white_noise"
            / f"{args.white_noise_dataset_size}-examples"
        )
        config["train_file"] = wn_dir / f"white_noise_obs_train_{dataset_idx}.npy"
        config["val_file"] = wn_dir / f"white_noise_obs_val_{dataset_idx}.npy"
        config["train_target_file"] = wn_dir / f"white_noise_output_train_{dataset_idx}.npy"
        config["val_target_file"] = wn_dir / f"white_noise_states_val_{dataset_idx}.npy"
        config["train_indices_file"] = wn_dir / f"white_noise_indices_train_{dataset_idx}.npy"
        config["val_indices_file"] = wn_dir / f"white_noise_indices_val_{dataset_idx}.npy"
        config["use_float_x"] = False
        config["use_float_y"] = config.get("output_vocab_size") is None

        save_ckpt_dir = (
            ckpt_dir / args.model_type / scaling_dir
            / f"next_token_pt_white_noise_idx_{dataset_idx}_transfer"
        )

        ddp, master_process, ptdtype, config = setup_training_environment(
            config, save_ckpt_dir, False
        )

        model, config, iter_num, _, best_val_loss, optimizer, scaler = init_model(
            config=config, ckpt_dir=pretrained_ckpt_dir, ddp=ddp
        )

        train(
            model=model, optimizer=optimizer, scaler=scaler, config=config,
            ddp=ddp, master_process=master_process, ptdtype=ptdtype,
            iter_num=0, current_epoch=0, best_val_loss=float("inf"),
            ckpt_dir=save_ckpt_dir, save_checkpoints=False,
        )

        # Save extrapolations
        ext_dir = domain_cfg["ext_dir"]
        if args.domain == "gridworld":
            ext_dir = ext_dir / f"{args.num_states}-states"
        ext_dir = ext_dir / "white_noise"
        ext_idx_dir = (
            ext_dir / args.model_type
            / f"pt_next_token"
            / f"scaling_{label}"
            / f"{args.white_noise_dataset_size}_examples"
            / f"{args.wn_max_iters}_iters"
            / f"idx_{dataset_idx}"
        )
        generate_and_save_extrapolations(model, config, ext_dir, ext_idx_dir)


def main():
    args = parse_args()
    domain_cfg = DOMAIN_CONFIGS[args.domain]
    model_configs = get_model_configs(args)

    if args.mode == "train_ntp":
        results = {}
        for n_layer, n_embd in model_configs:
            label = get_scale_label(n_layer, n_embd)
            val_loss = train_ntp_at_scale(args, n_layer, n_embd, domain_cfg)
            results[label] = {"n_layer": n_layer, "n_embd": n_embd,
                              "val_loss": float(val_loss)}

        # Save results
        results_dir = domain_cfg["ext_dir"]
        if args.domain == "gridworld":
            results_dir = results_dir / f"{args.num_states}-states"
        results_dir.mkdir(parents=True, exist_ok=True)
        with open(results_dir / f"scaling_ntp_results_{args.model_type}.json", "w") as f:
            json.dump(results, f, indent=4)

    elif args.mode == "train_wn":
        for n_layer, n_embd in model_configs:
            train_wn_at_scale(args, n_layer, n_embd, domain_cfg)

    elif args.mode == "compute_ib":
        # Compute IB for each scale (reuses existing compute_inductive_bias logic)
        logger.info("IB computation should be run using the domain-specific "
                     "compute_inductive_bias.py with appropriate paths.")


if __name__ == "__main__":
    main()
