#!/bin/bash
# =============================================================================
# Direction B: The Heuristic Trap — NTP + Contrastive Regularizer
# =============================================================================
# Trains models with contrastive representation regularizer at various lambda
# values, then compares IB metrics between baseline and regularized models.
#
# Usage:
#   bash scripts/run_direction_b.sh [--domain gridworld|heat_equation]
#                                    [--model_type gpt|mamba|rnn|lstm]
#                                    [--skip-data] [--skip-baseline]
#
# Prerequisites:
#   - Data must be generated (run Direction A step 0 first, or generate manually)
# =============================================================================

set -euo pipefail

# ---- Configuration ----
DOMAIN="gridworld"
MODEL_TYPE="gpt"
SKIP_DATA="false"
SKIP_BASELINE="false"

for arg in "$@"; do
    case $arg in
        --domain=*) DOMAIN="${arg#*=}" ;;
        --model_type=*) MODEL_TYPE="${arg#*=}" ;;
        --skip-data) SKIP_DATA="true" ;;
        --skip-baseline) SKIP_BASELINE="true" ;;
    esac
done

NUM_STATES=5
MAX_ITERS=60000
WN_MAX_ITERS=100
NUM_WN_DATASETS=100
WN_DATASET_SIZE=100

# Lambda grid for ablation
LAMBDAS=(0.001 0.01 0.1 0.5 1.0)

echo "============================================="
echo "Direction B: The Heuristic Trap"
echo "  Domain:     $DOMAIN"
echo "  Model:      $MODEL_TYPE"
echo "  Lambda grid: ${LAMBDAS[*]}"
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

# ---- Step 1: Train baseline NTP model (standard, no regularizer) ----
if [ "$SKIP_BASELINE" != "true" ]; then
    echo ""
    echo "[Step 1] Training baseline NTP model..."
    if [ "$DOMAIN" = "gridworld" ]; then
        python -m inductivebiasprobes.experiments.gridworld.train_model \
            --config ntp_config \
            --model_type $MODEL_TYPE \
            --max_iters $MAX_ITERS \
            --num_states $NUM_STATES \
            --no_wandb \
            --no_compile
    elif [ "$DOMAIN" = "heat_equation" ]; then
        python -m inductivebiasprobes.experiments.heat_equation.train_model \
            --config ntp_config \
            --model_type $MODEL_TYPE \
            --max_iters $MAX_ITERS \
            --no_wandb \
            --no_compile
    fi
    echo "[Step 1] Baseline training complete."

    # Train baseline white noise
    echo "[Step 1b] Training baseline white noise models..."
    if [ "$DOMAIN" = "gridworld" ]; then
        python -m inductivebiasprobes.experiments.gridworld.train_model \
            --config white_noise_config \
            --model_type $MODEL_TYPE \
            --pretrained next_token \
            --max_iters $WN_MAX_ITERS \
            --num_states $NUM_STATES \
            --num_white_noise_datasets $NUM_WN_DATASETS \
            --white_noise_dataset_size $WN_DATASET_SIZE \
            --no_wandb \
            --no_compile
    elif [ "$DOMAIN" = "heat_equation" ]; then
        python -m inductivebiasprobes.experiments.heat_equation.train_model \
            --config white_noise_config \
            --model_type $MODEL_TYPE \
            --pretrained next_token \
            --max_iters $WN_MAX_ITERS \
            --num_white_noise_datasets $NUM_WN_DATASETS \
            --white_noise_dataset_size $WN_DATASET_SIZE \
            --no_wandb \
            --no_compile
    fi
    echo "[Step 1b] Baseline white noise complete."
fi

# ---- Step 2: Train regularized models at each lambda ----
echo ""
echo "[Step 2] Training regularized models..."
for LAMBDA in "${LAMBDAS[@]}"; do
    echo "  Training with lambda=$LAMBDA..."

    # NTP + contrastive regularizer
    python -m inductivebiasprobes.experiments.direction_b.train_with_regularizer \
        --domain $DOMAIN \
        --model_type $MODEL_TYPE \
        --lambda_contrast $LAMBDA \
        --mode ntp \
        --max_iters $MAX_ITERS \
        --num_states $NUM_STATES \
        --no_wandb \
        --no_compile

    # White noise fine-tuning from regularized checkpoint
    python -m inductivebiasprobes.experiments.direction_b.train_with_regularizer \
        --domain $DOMAIN \
        --model_type $MODEL_TYPE \
        --lambda_contrast $LAMBDA \
        --mode white_noise \
        --max_iters $WN_MAX_ITERS \
        --num_states $NUM_STATES \
        --num_white_noise_datasets $NUM_WN_DATASETS \
        --white_noise_dataset_size $WN_DATASET_SIZE \
        --no_wandb \
        --no_compile

    echo "  Done with lambda=$LAMBDA"
done
echo "[Step 2] Regularized training complete."

# ---- Step 3: Compute IB for baseline ----
echo ""
echo "[Step 3] Computing baseline IB..."
if [ "$DOMAIN" = "gridworld" ]; then
    python -m inductivebiasprobes.experiments.gridworld.compute_inductive_bias \
        --num_states $NUM_STATES \
        --model_type $MODEL_TYPE \
        --pretrained next_token \
        --white_noise_dataset_size $WN_DATASET_SIZE \
        --num_white_noise_datasets $NUM_WN_DATASETS \
        --max_iters $WN_MAX_ITERS
    BASELINE_IB="inductivebiasprobes/extrapolations/gridworld/${NUM_STATES}-states/white_noise/${MODEL_TYPE}/pt_next_token/${WN_DATASET_SIZE}_examples/${WN_MAX_ITERS}_iters/ib.json"
elif [ "$DOMAIN" = "heat_equation" ]; then
    python -m inductivebiasprobes.experiments.heat_equation.compute_inductive_bias \
        --model_type $MODEL_TYPE \
        --pretrained next_token \
        --white_noise_dataset_size $WN_DATASET_SIZE \
        --num_white_noise_datasets $NUM_WN_DATASETS \
        --max_iters $WN_MAX_ITERS
    BASELINE_IB="inductivebiasprobes/extrapolations/heat_equation/white_noise/${MODEL_TYPE}/pt_next_token/${WN_DATASET_SIZE}_examples/${WN_MAX_ITERS}_iters/ib.json"
fi
echo "[Step 3] Baseline IB computed."

# ---- Step 4: Compute IB for each regularized model ----
echo ""
echo "[Step 4] Computing regularized IB metrics..."
REG_IB_FILES=""
LAMBDA_ARGS=""
for LAMBDA in "${LAMBDAS[@]}"; do
    echo "  Computing IB for lambda=$LAMBDA..."
    if [ "$DOMAIN" = "gridworld" ]; then
        EXT_DIR="inductivebiasprobes/extrapolations/gridworld/${NUM_STATES}-states/white_noise"
        REG_IB="${EXT_DIR}/${MODEL_TYPE}/pt_regularized_lambda_${LAMBDA}/${WN_DATASET_SIZE}_examples/${WN_MAX_ITERS}_iters/ib.json"
    elif [ "$DOMAIN" = "heat_equation" ]; then
        EXT_DIR="inductivebiasprobes/extrapolations/heat_equation/white_noise"
        REG_IB="${EXT_DIR}/${MODEL_TYPE}/pt_regularized_lambda_${LAMBDA}/${WN_DATASET_SIZE}_examples/${WN_MAX_ITERS}_iters/ib.json"
    fi
    REG_IB_FILES="$REG_IB_FILES $REG_IB"
    LAMBDA_ARGS="$LAMBDA_ARGS $LAMBDA"
done
echo "[Step 4] Regularized IB computed."

# ---- Step 5: Compare and plot ----
echo ""
echo "[Step 5] Generating comparison plots..."
if [ "$DOMAIN" = "gridworld" ]; then
    OUTPUT_DIR="inductivebiasprobes/extrapolations/gridworld/${NUM_STATES}-states/direction_b_figures"
elif [ "$DOMAIN" = "heat_equation" ]; then
    OUTPUT_DIR="inductivebiasprobes/extrapolations/heat_equation/direction_b_figures"
fi

python -m inductivebiasprobes.experiments.direction_b.compare_ib \
    --baseline_ib "$BASELINE_IB" \
    --regularized_ibs $REG_IB_FILES \
    --lambda_values $LAMBDA_ARGS \
    --output_dir "$OUTPUT_DIR"

echo ""
echo "============================================="
echo "Direction B complete!"
echo "  Figures: $OUTPUT_DIR/"
echo "============================================="
