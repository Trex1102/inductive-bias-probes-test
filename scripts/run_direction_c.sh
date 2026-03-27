#!/bin/bash
# =============================================================================
# Direction C: IB Dynamics as a Reliability Diagnostic
# =============================================================================
# Tracks IB metrics throughout NTP pretraining using log-spaced checkpoints.
# Shows that IB predicts transfer better than loss alone.
#
# Usage:
#   bash scripts/run_direction_c.sh [--domain gridworld|heat_equation]
#                                    [--model_type gpt|mamba|rnn|lstm]
#                                    [--skip-data] [--skip-train]
#
# Prerequisites:
#   - Data must be generated (run generate_data.py and generate_white_noise.py)
# =============================================================================

set -euo pipefail

# ---- Configuration ----
DOMAIN="gridworld"
MODEL_TYPE="gpt"
SKIP_DATA="false"
SKIP_TRAIN="false"

for arg in "$@"; do
    case $arg in
        --domain=*) DOMAIN="${arg#*=}" ;;
        --model_type=*) MODEL_TYPE="${arg#*=}" ;;
        --skip-data) SKIP_DATA="true" ;;
        --skip-train) SKIP_TRAIN="true" ;;
    esac
done

NUM_STATES=5
MAX_ITERS=60000
WN_MAX_ITERS=100
NUM_WN_DATASETS=20   # Fewer WN datasets per checkpoint (speed vs accuracy tradeoff)
WN_DATASET_SIZE=100

echo "============================================="
echo "Direction C: IB Dynamics Diagnostic"
echo "  Domain:     $DOMAIN"
echo "  Model:      $MODEL_TYPE"
echo "  Max iters:  $MAX_ITERS"
echo "  WN datasets per checkpoint: $NUM_WN_DATASETS"
echo "============================================="

# ---- Step 0: Generate data (if needed) ----
if [ "$SKIP_DATA" != "true" ]; then
    echo ""
    echo "[Step 0] Generating data..."
    if [ "$DOMAIN" = "gridworld" ]; then
        python -m inductivebiasprobes.experiments.gridworld.generate_data
        python -m inductivebiasprobes.experiments.gridworld.generate_white_noise \
            --num_white_noise_datasets $NUM_WN_DATASETS \
            --white_noise_dataset_size $WN_DATASET_SIZE
    elif [ "$DOMAIN" = "heat_equation" ]; then
        python -m inductivebiasprobes.experiments.heat_equation.generate_data
        python -m inductivebiasprobes.experiments.heat_equation.generate_white_noise \
            --num_white_noise_datasets $NUM_WN_DATASETS \
            --white_noise_dataset_size $WN_DATASET_SIZE
    fi
    echo "[Step 0] Data generation complete."
fi

# ---- Step 1: Train NTP model with log-spaced checkpoint saving ----
if [ "$SKIP_TRAIN" != "true" ]; then
    echo ""
    echo "[Step 1] Training NTP model with log-spaced checkpoints..."
    python -m inductivebiasprobes.experiments.direction_c.train_with_checkpoints \
        --domain $DOMAIN \
        --model_type $MODEL_TYPE \
        --max_iters $MAX_ITERS \
        --num_states $NUM_STATES \
        --no_wandb
    echo "[Step 1] Training complete."
fi

# ---- Step 2: Compute IB at each checkpoint ----
echo ""
echo "[Step 2] Computing IB dynamics across checkpoints..."
echo "  (This is the most compute-intensive step: ~$NUM_WN_DATASETS WN runs per checkpoint)"
python -m inductivebiasprobes.experiments.direction_c.compute_ib_dynamics \
    --domain $DOMAIN \
    --model_type $MODEL_TYPE \
    --num_states $NUM_STATES \
    --max_iters $MAX_ITERS \
    --wn_max_iters $WN_MAX_ITERS \
    --num_white_noise_datasets $NUM_WN_DATASETS \
    --white_noise_dataset_size $WN_DATASET_SIZE
echo "[Step 2] IB dynamics computation complete."

# ---- Step 3: Generate plots and analysis ----
echo ""
echo "[Step 3] Generating IB dynamics plots..."
if [ "$DOMAIN" = "gridworld" ]; then
    EXT_DIR="inductivebiasprobes/extrapolations/gridworld/${NUM_STATES}-states"
elif [ "$DOMAIN" = "heat_equation" ]; then
    EXT_DIR="inductivebiasprobes/extrapolations/heat_equation"
fi

DYNAMICS_FILE="${EXT_DIR}/ib_dynamics/${MODEL_TYPE}/dynamics/ib_dynamics.json"
OUTPUT_DIR="${EXT_DIR}/direction_c_figures"

python -m inductivebiasprobes.experiments.direction_c.compute_transfer_correlation \
    --dynamics_file "$DYNAMICS_FILE" \
    --output_dir "$OUTPUT_DIR"

echo ""
echo "============================================="
echo "Direction C complete!"
echo "  IB dynamics: $DYNAMICS_FILE"
echo "  Figures:     $OUTPUT_DIR/"
echo "============================================="
echo ""
echo "Key outputs:"
echo "  - ib_dynamics.pdf: IB metrics vs training step (hero figure)"
echo "  - ib_vs_loss.pdf: IB vs NTP loss disentanglement"
echo "  - ib_plateau.pdf: IB change rate for early stopping detection"
echo "  - ib_dynamics_summary.json: Correlation statistics"
