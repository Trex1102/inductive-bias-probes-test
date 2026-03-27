"""Direction B: Train models with contrastive representation regularizer.

Trains NTP models with an additional contrastive loss term:
    L_total = L_NTP + lambda * L_contrast

The contrastive loss encourages hidden representations to preserve pairwise
distances from the input space, preventing collapse to next-token equivalence
classes (the "heuristic trap").
"""

import argparse
import logging

import torch
import wandb
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
from inductivebiasprobes.src.contrastive_regularizer import (
    create_contrastive_loss_fn,
    get_log_spaced_steps,
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

DOMAIN_CONFIGS = {
    "gridworld": {
        "ckpt_dir": GRIDWORLD_CKPT_DIR,
        "config_dir": GRIDWORLD_CONFIG_DIR,
        "data_dir": GRIDWORLD_DATA_DIR,
        "ext_dir": GRIDWORLD_EXT_DIR,
    },
    "heat_equation": {
        "ckpt_dir": HEAT_EQ_CKPT_DIR,
        "config_dir": HEAT_EQ_CONFIG_DIR,
        "data_dir": HEAT_EQ_DATA_DIR,
        "ext_dir": HEAT_EQ_EXT_DIR,
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Direction B: Train with contrastive regularizer"
    )
    parser = add_common_args(parser)
    parser.add_argument("--domain", type=str, required=True,
                        choices=["gridworld", "heat_equation"])
    parser.add_argument("--lambda_contrast", type=float, required=True,
                        help="Weight of contrastive loss (e.g., 0.01)")
    parser.add_argument("--num_states", type=int, default=5,
                        help="For gridworld: number of states")
    parser.add_argument("--mode", type=str, default="ntp",
                        choices=["ntp", "white_noise"],
                        help="ntp: pretrain with regularizer, "
                             "white_noise: fine-tune on white noise")
    return parser.parse_args()


def load_ntp_config(args, domain_cfg):
    """Load NTP config for the domain."""
    if args.domain == "gridworld":
        config_path = (
            domain_cfg["config_dir"] / f"{args.num_states}-states" / "ntp_config.yaml"
        )
    else:
        config_path = domain_cfg["config_dir"] / "ntp_config.yaml"
    with open(config_path) as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def load_wn_config(args, domain_cfg):
    """Load white noise config for the domain."""
    if args.domain == "gridworld":
        config_path = (
            domain_cfg["config_dir"]
            / f"{args.num_states}-states"
            / "white_noise_config.yaml"
        )
    else:
        config_path = domain_cfg["config_dir"] / "white_noise_config.yaml"
    with open(config_path) as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def build_config(args, base_config):
    """Merge CLI args into config."""
    config = dict(base_config)
    cli = vars(args)
    for k, v in cli.items():
        if v is not None and k in config:
            config[k] = v
    # Set defaults for missing training params
    config.setdefault("no_wandb", True)
    config.setdefault("no_compile", True)  # Required for return_reps
    config.setdefault("device", args.device)
    config.setdefault("dtype", args.dtype)
    config.setdefault("eval_interval", 500)
    config.setdefault("eval_iters", 10)
    config.setdefault("batch_size", 128)
    config.setdefault("learning_rate", 6e-4)
    config.setdefault("weight_decay", 0.1)
    config.setdefault("beta1", 0.9)
    config.setdefault("beta2", 0.95)
    config.setdefault("grad_clip", 1.0)
    config.setdefault("decay_lr", False)
    config.setdefault("warmup_iters", 2000)
    config.setdefault("lr_decay_iters", config.get("max_iters", 60000))
    config.setdefault("min_lr", 6e-5)
    config.setdefault("log_interval", 100)
    config.setdefault("always_save_checkpoint", False)
    config.setdefault("backend", "nccl")
    config.setdefault("save_loss", False)
    config.setdefault("resume_from_last_ckpt", False)
    config.setdefault("plot_trajectory", False)
    config.setdefault("gradient_accumulation_steps", 1)

    # Model type defaults
    config["model_type"] = args.model_type
    if args.model_type == "gpt":
        config.setdefault("n_head", max(1, config.get("n_embd", 768) // 64))
        config.setdefault("dropout", 0.0)
        for k in ["dt_rank", "d_state", "expand_factor", "d_conv", "dt_min",
                   "dt_max", "dt_init", "dt_scale", "dt_init_floor",
                   "rms_norm_eps", "conv_bias", "inner_layernorms", "pscan", "use_cuda"]:
            config[k] = None
    elif args.model_type in ("mamba", "mamba2", "rnn", "lstm"):
        config.update({
            "dt_rank": "auto", "d_state": 16, "expand_factor": 2,
            "d_conv": 4, "dt_min": 0.001, "dt_max": 0.1,
            "dt_init": "random", "dt_scale": 1.0, "dt_init_floor": 1e-4,
            "rms_norm_eps": 1e-5, "conv_bias": True,
            "inner_layernorms": True, "pscan": True, "use_cuda": True,
            "n_head": None, "dropout": None,
        })

    return config


def train_ntp_with_regularizer(args):
    """Train NTP model with contrastive regularizer."""
    domain_cfg = DOMAIN_CONFIGS[args.domain]
    base_config = load_ntp_config(args, domain_cfg)
    config = build_config(args, base_config)
    config["pretrained"] = "scratch"

    # IMPORTANT: disable compile for return_reps support
    config["no_compile"] = True

    lambda_str = f"lambda_{args.lambda_contrast}"

    # Data paths
    if args.domain == "gridworld":
        data_dir = domain_cfg["data_dir"] / f"{args.num_states}-states"
        ckpt_base = domain_cfg["ckpt_dir"] / f"{args.num_states}-states"
    else:
        data_dir = domain_cfg["data_dir"]
        ckpt_base = domain_cfg["ckpt_dir"]

    config["train_file"] = data_dir / "obs_train.npy"
    config["val_file"] = data_dir / "obs_val.npy"
    config["use_float_x"] = False
    config["use_float_y"] = config.get("output_vocab_size") is None

    save_ckpt_dir = (
        ckpt_base / config["model_type"] / f"regularized_{lambda_str}" / "next_token"
    )

    # Create contrastive loss function
    extra_loss_fn = create_contrastive_loss_fn(args.lambda_contrast)

    # Log-spaced checkpoint saving for IB dynamics tracking
    checkpoint_steps = get_log_spaced_steps(config.get("max_iters", 60000))

    ddp, master_process, ptdtype, config = setup_training_environment(
        config, save_ckpt_dir, True
    )

    model, config, iter_num, current_epoch, best_val_loss, optimizer, scaler = (
        init_model(config=config, ckpt_dir=save_ckpt_dir, ddp=ddp)
    )

    logger.info(f"Training with contrastive regularizer: lambda={args.lambda_contrast}")
    logger.info(f"Saving {len(checkpoint_steps)} log-spaced checkpoints")

    train(
        model=model, optimizer=optimizer, scaler=scaler, config=config,
        ddp=ddp, master_process=master_process, ptdtype=ptdtype,
        iter_num=iter_num, current_epoch=current_epoch,
        best_val_loss=best_val_loss, ckpt_dir=save_ckpt_dir,
        save_checkpoints=True,
        save_checkpoint_steps=checkpoint_steps,
        extra_loss_fn=extra_loss_fn,
    )

    logger.info(f"Done. Checkpoints saved to {save_ckpt_dir}")


def train_wn_from_regularized(args):
    """Train white noise models from a regularized pretrained checkpoint."""
    domain_cfg = DOMAIN_CONFIGS[args.domain]
    wn_config = load_wn_config(args, domain_cfg)
    config = build_config(args, wn_config)

    lambda_str = f"lambda_{args.lambda_contrast}"

    if args.domain == "gridworld":
        data_dir = domain_cfg["data_dir"] / f"{args.num_states}-states"
        ckpt_base = domain_cfg["ckpt_dir"] / f"{args.num_states}-states"
        ext_base = domain_cfg["ext_dir"] / f"{args.num_states}-states"
    else:
        data_dir = domain_cfg["data_dir"]
        ckpt_base = domain_cfg["ckpt_dir"]
        ext_base = domain_cfg["ext_dir"]

    pretrained_ckpt_dir = (
        ckpt_base / config["model_type"] / f"regularized_{lambda_str}" / "next_token"
    )
    config["pretrained"] = "next_token"

    wn_dataset_size = config.get("num_data_points", 100)
    num_wn = args.num_white_noise_datasets if hasattr(args, "num_white_noise_datasets") else 100
    wn_max_iters = config.get("max_iters", 100)

    for idx in range(num_wn):
        logger.info(f"WN dataset {idx}/{num_wn} (lambda={args.lambda_contrast})")

        cfg = dict(config)
        wn_dir = data_dir / "white_noise" / f"{wn_dataset_size}-examples"
        cfg["train_file"] = wn_dir / f"white_noise_obs_train_{idx}.npy"
        cfg["val_file"] = wn_dir / f"white_noise_obs_val_{idx}.npy"
        cfg["train_target_file"] = wn_dir / f"white_noise_output_train_{idx}.npy"
        cfg["val_target_file"] = wn_dir / f"white_noise_states_val_{idx}.npy"
        cfg["train_indices_file"] = wn_dir / f"white_noise_indices_train_{idx}.npy"
        cfg["val_indices_file"] = wn_dir / f"white_noise_indices_val_{idx}.npy"
        cfg["use_float_x"] = False
        cfg["use_float_y"] = cfg.get("output_vocab_size") is None
        cfg["no_wandb"] = True
        cfg["no_compile"] = True

        save_ckpt_dir = (
            ckpt_base / config["model_type"] / f"regularized_{lambda_str}"
            / f"next_token_pt_white_noise_idx_{idx}_transfer"
        )

        ddp, master_process, ptdtype, cfg = setup_training_environment(
            cfg, save_ckpt_dir, False
        )
        model, cfg, _, _, _, optimizer, scaler = init_model(
            config=cfg, ckpt_dir=pretrained_ckpt_dir, ddp=ddp
        )

        train(
            model=model, optimizer=optimizer, scaler=scaler, config=cfg,
            ddp=ddp, master_process=master_process, ptdtype=ptdtype,
            iter_num=0, current_epoch=0, best_val_loss=float("inf"),
            ckpt_dir=save_ckpt_dir, save_checkpoints=False,
        )

        # Save extrapolations
        ext_dir = ext_base / "white_noise"
        ext_idx_dir = (
            ext_dir / config["model_type"]
            / f"pt_regularized_{lambda_str}"
            / f"{wn_dataset_size}_examples"
            / f"{wn_max_iters}_iters"
            / f"idx_{idx}"
        )
        generate_and_save_extrapolations(model, cfg, ext_dir, ext_idx_dir)


def main():
    args = parse_args()
    if args.mode == "ntp":
        train_ntp_with_regularizer(args)
    elif args.mode == "white_noise":
        train_wn_from_regularized(args)


if __name__ == "__main__":
    main()
