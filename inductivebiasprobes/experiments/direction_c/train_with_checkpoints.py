"""Direction C: Train models with log-spaced checkpoint saving.

Saves checkpoints at log-spaced intervals during NTP pretraining
to enable tracking IB dynamics throughout training.
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
    HEAT_EQ_CKPT_DIR,
    HEAT_EQ_CONFIG_DIR,
    HEAT_EQ_DATA_DIR,
)
from inductivebiasprobes.src.contrastive_regularizer import get_log_spaced_steps
from inductivebiasprobes.src.train_utils import (
    add_common_args,
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
    },
    "heat_equation": {
        "ckpt_dir": HEAT_EQ_CKPT_DIR,
        "config_dir": HEAT_EQ_CONFIG_DIR,
        "data_dir": HEAT_EQ_DATA_DIR,
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Direction C: Train with log-spaced checkpoints"
    )
    parser = add_common_args(parser)
    parser.add_argument("--domain", type=str, required=True,
                        choices=["gridworld", "heat_equation"])
    parser.add_argument("--num_states", type=int, default=5,
                        help="For gridworld: number of states")
    return parser.parse_args()


def load_config(args, domain_cfg):
    """Load NTP config."""
    if args.domain == "gridworld":
        config_path = (
            domain_cfg["config_dir"] / f"{args.num_states}-states" / "ntp_config.yaml"
        )
    else:
        config_path = domain_cfg["config_dir"] / "ntp_config.yaml"

    with open(config_path) as f:
        base = yaml.load(f, Loader=yaml.FullLoader)

    cli = vars(args)
    for k, v in cli.items():
        if v is not None and k in base:
            base[k] = v

    base.setdefault("no_wandb", True)
    base.setdefault("no_compile", False)
    base.setdefault("device", args.device)
    base.setdefault("dtype", args.dtype)
    base.setdefault("eval_interval", 500)
    base.setdefault("eval_iters", 10)
    base.setdefault("batch_size", 128)
    base.setdefault("learning_rate", 6e-4)
    base.setdefault("weight_decay", 0.1)
    base.setdefault("beta1", 0.9)
    base.setdefault("beta2", 0.95)
    base.setdefault("grad_clip", 1.0)
    base.setdefault("decay_lr", False)
    base.setdefault("warmup_iters", 2000)
    base.setdefault("lr_decay_iters", base.get("max_iters", 60000))
    base.setdefault("min_lr", 6e-5)
    base.setdefault("log_interval", 100)
    base.setdefault("always_save_checkpoint", False)
    base.setdefault("backend", "nccl")
    base.setdefault("save_loss", False)
    base.setdefault("resume_from_last_ckpt", False)
    base.setdefault("plot_trajectory", False)
    base.setdefault("gradient_accumulation_steps", 1)
    base["pretrained"] = "scratch"
    base["predict_type"] = "next_token"
    base["model_type"] = args.model_type

    # Set architecture-specific defaults
    if args.model_type == "gpt":
        base.setdefault("n_head", max(1, base.get("n_embd", 768) // 64))
        base.setdefault("dropout", 0.0)
        for k in ["dt_rank", "d_state", "expand_factor", "d_conv", "dt_min",
                   "dt_max", "dt_init", "dt_scale", "dt_init_floor",
                   "rms_norm_eps", "conv_bias", "inner_layernorms", "pscan", "use_cuda"]:
            base[k] = None
    elif args.model_type in ("mamba", "mamba2", "rnn", "lstm"):
        base.update({
            "dt_rank": "auto", "d_state": 16, "expand_factor": 2,
            "d_conv": 4, "dt_min": 0.001, "dt_max": 0.1,
            "dt_init": "random", "dt_scale": 1.0, "dt_init_floor": 1e-4,
            "rms_norm_eps": 1e-5, "conv_bias": True,
            "inner_layernorms": True, "pscan": True, "use_cuda": True,
            "n_head": None, "dropout": None,
        })
        if args.model_type in ("rnn", "lstm"):
            base["n_layer"] = 2

    return base


def main():
    args = parse_args()
    domain_cfg = DOMAIN_CONFIGS[args.domain]
    config = load_config(args, domain_cfg)

    max_iters = config.get("max_iters", 60000)
    checkpoint_steps = get_log_spaced_steps(max_iters)
    logger.info(f"Will save {len(checkpoint_steps)} log-spaced checkpoints: "
                f"{checkpoint_steps[:5]}...{checkpoint_steps[-3:]}")

    # Data paths
    if args.domain == "gridworld":
        data_dir = domain_cfg["data_dir"] / f"{config.get('num_states', 5)}-states"
        ckpt_base = domain_cfg["ckpt_dir"] / f"{config.get('num_states', 5)}-states"
    else:
        data_dir = domain_cfg["data_dir"]
        ckpt_base = domain_cfg["ckpt_dir"]

    config["train_file"] = data_dir / "obs_train.npy"
    config["val_file"] = data_dir / "obs_val.npy"
    config["use_float_x"] = False
    config["use_float_y"] = config.get("output_vocab_size") is None

    save_ckpt_dir = ckpt_base / config["model_type"] / "dynamics" / "next_token"

    ddp, master_process, ptdtype, config = setup_training_environment(
        config, save_ckpt_dir, True
    )

    model, config, iter_num, current_epoch, best_val_loss, optimizer, scaler = (
        init_model(config=config, ckpt_dir=save_ckpt_dir, ddp=ddp)
    )

    def misclassification_callback(output, targets):
        top_preds = torch.argmax(output, dim=-1).view(-1)
        targets = targets.view(-1)
        return (top_preds != targets).float().mean()

    train(
        model=model, optimizer=optimizer, scaler=scaler, config=config,
        ddp=ddp, master_process=master_process, ptdtype=ptdtype,
        iter_num=iter_num, current_epoch=current_epoch,
        best_val_loss=best_val_loss, ckpt_dir=save_ckpt_dir,
        save_checkpoints=True,
        save_checkpoint_steps=checkpoint_steps,
        loss_callback=misclassification_callback,
        loss_callback_name="miscl_rate",
    )

    logger.info(f"Done. Checkpoints saved to {save_ckpt_dir}")
    logger.info(f"Checkpoint steps: {checkpoint_steps}")


if __name__ == "__main__":
    main()
