#!/bin/bash
# =============================================================================
# Direction A: Scaling Laws for World Model Quality
# =============================================================================
# Measures how IB metrics scale with model size and compares to loss scaling.
#
# Usage:
#   bash scripts/run_direction_a.sh [--domain gridworld|heat_equation]
#                                    [--model_type gpt|mamba|rnn|lstm]
#                                    [--skip-data] [--skip-ntp] [--skip-wn]
#
# Prerequisites:
#   - For gridworld: run generate_data.py and generate_white_noise.py first
#   - For heat_equation: run heat_equation/generate_data.py and generate_white_noise.py first
# =============================================================================

set -euo pipefail

# ---- Configuration ----
DOMAIN="${1:-gridworld}"
MODEL_TYPE="${2:-gpt}"
SKIP_DATA="${3:-false}"
SKIP_NTP="${4:-false}"
SKIP_WN="${5:-false}"

# Parse named arguments
for arg in "$@"; do
    case $arg in
        --domain=*) DOMAIN="${arg#*=}" ;;
        --model_type=*) MODEL_TYPE="${arg#*=}" ;;
        --skip-data) SKIP_DATA="true" ;;
        --skip-ntp) SKIP_NTP="true" ;;
        --skip-wn) SKIP_WN="true" ;;
    esac
done

NUM_STATES=5
MAX_ITERS=60000
WN_MAX_ITERS=100
NUM_WN_DATASETS=100
WN_DATASET_SIZE=100

# Model size grid (subset for feasibility on single GPU)
LAYERS=(2 4 6 8)
EMBEDS=(64 128 256)

echo "============================================="
echo "Direction A: IB Scaling Laws"
echo "  Domain:     $DOMAIN"
echo "  Model:      $MODEL_TYPE"
echo "  Layer grid: ${LAYERS[*]}"
echo "  Embed grid: ${EMBEDS[*]}"
echo "============================================="

# ---- Step 0: Generate data (if needed) ----
if [ "$SKIP_DATA" != "true" ]; then
    echo ""
    echo "[Step 0] Generating data for $DOMAIN..."
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

# ---- Step 1: Train NTP models at each scale ----
if [ "$SKIP_NTP" != "true" ]; then
    echo ""
    echo "[Step 1] Training NTP models at each scale..."
    for n_layer in "${LAYERS[@]}"; do
        for n_embd in "${EMBEDS[@]}"; do
            echo "  Training L${n_layer}_D${n_embd}..."
            python -m inductivebiasprobes.experiments.direction_a.train_scaling \
                --domain $DOMAIN \
                --model_type $MODEL_TYPE \
                --mode train_ntp \
                --n_layer $n_layer \
                --n_embd $n_embd \
                --max_iters $MAX_ITERS \
                --num_states $NUM_STATES
        done
    done
    echo "[Step 1] NTP training complete."
fi

# ---- Step 2: Train white noise models for each scale ----
if [ "$SKIP_WN" != "true" ]; then
    echo ""
    echo "[Step 2] Training white noise models for IB probes..."
    for n_layer in "${LAYERS[@]}"; do
        for n_embd in "${EMBEDS[@]}"; do
            echo "  WN for L${n_layer}_D${n_embd}..."
            python -m inductivebiasprobes.experiments.direction_a.train_scaling \
                --domain $DOMAIN \
                --model_type $MODEL_TYPE \
                --mode train_wn \
                --n_layer $n_layer \
                --n_embd $n_embd \
                --max_iters $MAX_ITERS \
                --wn_max_iters $WN_MAX_ITERS \
                --num_white_noise_datasets $NUM_WN_DATASETS \
                --white_noise_dataset_size $WN_DATASET_SIZE \
                --num_states $NUM_STATES
        done
    done
    echo "[Step 2] White noise training complete."
fi

# ---- Step 3: Compute IB metrics at each scale ----
echo ""
echo "[Step 3] Computing IB metrics..."
if [ "$DOMAIN" = "gridworld" ]; then
    RESULTS_DIR="inductivebiasprobes/extrapolations/gridworld/${NUM_STATES}-states"
    for n_layer in "${LAYERS[@]}"; do
        for n_embd in "${EMBEDS[@]}"; do
            SCALE="scaling_L${n_layer}_D${n_embd}"
            echo "  IB for ${SCALE}..."
            python -m inductivebiasprobes.experiments.gridworld.compute_inductive_bias \
                --num_states $NUM_STATES \
                --model_type $MODEL_TYPE \
                --pretrained next_token \
                --white_noise_dataset_size $WN_DATASET_SIZE \
                --num_white_noise_datasets $NUM_WN_DATASETS \
                --max_iters $WN_MAX_ITERS || echo "  Warning: IB computation failed for ${SCALE}"
        done
    done
elif [ "$DOMAIN" = "heat_equation" ]; then
    RESULTS_DIR="inductivebiasprobes/extrapolations/heat_equation"
    for n_layer in "${LAYERS[@]}"; do
        for n_embd in "${EMBEDS[@]}"; do
            SCALE="scaling_L${n_layer}_D${n_embd}"
            echo "  IB for ${SCALE}..."
            python -m inductivebiasprobes.experiments.heat_equation.compute_inductive_bias \
                --model_type $MODEL_TYPE \
                --pretrained next_token \
                --white_noise_dataset_size $WN_DATASET_SIZE \
                --num_white_noise_datasets $NUM_WN_DATASETS \
                --max_iters $WN_MAX_ITERS || echo "  Warning: IB computation failed for ${SCALE}"
        done
    done
fi
echo "[Step 3] IB computation complete."

# ---- Step 4: Fit scaling laws and generate plots ----
echo ""
echo "[Step 4] Fitting scaling laws and generating plots..."
python -m inductivebiasprobes.experiments.direction_a.compute_scaling_laws \
    --results_dir "$RESULTS_DIR" \
    --model_type $MODEL_TYPE \
    --output_file "scaling_laws_${MODEL_TYPE}.json"

python -m inductivebiasprobes.experiments.direction_a.plot_scaling_laws \
    --results_file "${RESULTS_DIR}/scaling_laws_${MODEL_TYPE}.json" \
    --output_dir "${RESULTS_DIR}/figures"

echo ""
echo "============================================="
echo "Direction A complete!"
echo "  Results: ${RESULTS_DIR}/scaling_laws_${MODEL_TYPE}.json"
echo "  Figures: ${RESULTS_DIR}/figures/"
echo "============================================="
