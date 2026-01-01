#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
DATA_DIR="${PROJECT_ROOT}/dataset"

TRAIN_FILE=${TRAIN_FILE:-emolit12_labels_train.csv}
VALID_FILE=${VALID_FILE:-emolit12_labels_test.csv}
TEST_FILE=${TEST_FILE:-emolit12_labels_test.csv}
MODEL_NAME=${MODEL_NAME:-intfloat/multilingual-e5-large}
MODEL_ID=${MODEL_ID:-intfloat_multilingual-e5-large}
POOLER_TYPE=${POOLER_TYPE:-cls}
MAX_SEQ_LEN=${MAX_SEQ_LENGTH:-512}
LEARNING_RATE=${LR:-5e-5}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-128}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-128}
NUM_EPOCHS=${NUM_EPOCHS:-15}
GRAD_ACCUM=${GRADIENT_ACCUMULATION_STEPS:-1}
SEED=${SEED:-42}
MARGIN=${MARGIN:-0.0}
FP16=${FP16:-false}
BF16=${BF16:-true}
FREEZE_ENCODER=${FREEZE_ENCODER:-false}
CLASSIFIER_TYPE=${CLASSIFIER_TYPE:-nGPT}
LOSS_FUNCTION=${LOSS_FUNCTION:-CircularCSE}
LOG_OF_SUM=${LOG_OF_SUM:-false}
USE_ANGLE_MAP=${USE_ANGLE_MAP:-true}
INPUT_DIM=${INPUT_DIM:-1024}

OUTPUT_DIR="${PROJECT_ROOT}/outputs/${MODEL_ID}/${LOSS_FUNCTION}-${CLASSIFIER_TYPE}"
mkdir -p "${OUTPUT_DIR}"

EXPERIMENT_NAME=${EXPERIMENT_NAME:-Emolit12}
mkdir -p "${OUTPUT_DIR}/${EXPERIMENT_NAME}"

CONFIG_PATH="${OUTPUT_DIR}/${EXPERIMENT_NAME}/classifier_config.json"

# Generate config based on CLASSIFIER_TYPE and LOSS_FUNCTION
if [ "${LOSS_FUNCTION}" = "CircularCSE" ]; then
  cat <<JSON >"${CONFIG_PATH}"
{
  "sentiment": {
    "type": "${CLASSIFIER_TYPE}",
    "head_name": "sentiment_${CLASSIFIER_TYPE}_circular",
    "objective": "CircularCSE",
    "distance": "cosine",
    "layer": -1,
    "input_dim": ${INPUT_DIM},
    "pooler_type": "${POOLER_TYPE}",
    "margin": ${MARGIN},
    "angle_map": {
      "0": "love", "30": "joy", "60": "excitement",
      "90": "surprise", "120": "anger", "150": "fear",
      "180": "disgust", "210": "sadness", "240": "boredom",
      "270": "calmness", "300": "relief", "330": "trust"
    }
  }
}
JSON
elif [ "${LOSS_FUNCTION}" = "SINCERE" ]; then
  cat <<JSON >"${CONFIG_PATH}"
{
  "sentiment": {
    "type": "${CLASSIFIER_TYPE}",
    "head_name": "sentiment_${CLASSIFIER_TYPE}_sincere",
    "objective": "SINCERE",
    "distance": "cosine",
    "layer": -1,
    "input_dim": ${INPUT_DIM},
    "pooler_type": "${POOLER_TYPE}",
    "log_of_sum": ${LOG_OF_SUM},
    "tau": 0.05,
    "angle_map": {
      "0": "love", "30": "joy", "60": "excitement",
      "90": "surprise", "120": "anger", "150": "fear",
      "180": "disgust", "210": "sadness", "240": "boredom",
      "270": "calmness", "300": "relief", "330": "trust"
    }
  }
}
JSON
elif [ "${LOSS_FUNCTION}" = "SoftCSE" ]; then
  cat <<JSON >"${CONFIG_PATH}"
{
  "sentiment": {
    "type": "${CLASSIFIER_TYPE}",
    "head_name": "sentiment_${CLASSIFIER_TYPE}_softcse",
    "objective": "SoftCSE",
    "distance": "cosine",
    "layer": -1,
    "dropout": 0.1,
    "pooler_type": "${POOLER_TYPE}",
    "input_dim": ${INPUT_DIM},
    "log_of_sum": ${LOG_OF_SUM},
    "tau": 0.05,
    "use_angle_map": ${USE_ANGLE_MAP},
    "similarity_calculator": "intfloat/multilingual-e5-large",
    "angle_map": {
      "0": "love", "30": "joy", "60": "excitement",
      "90": "surprise", "120": "anger", "150": "fear",
      "180": "disgust", "210": "sadness", "240": "boredom",
      "270": "calmness", "300": "relief", "330": "trust"
    }
  }
}
JSON
else
  echo "Unknown LOSS_FUNCTION: ${LOSS_FUNCTION}" >&2
  exit 1
fi

CUDA_VISIBLE_DEVICES=0 python3 "${SCRIPT_DIR}/src/train.py" \
  --model_name_or_path "${MODEL_NAME}" \
  --output_dir "${OUTPUT_DIR}/${EXPERIMENT_NAME}" \
  --train_file "${DATA_DIR}/${TRAIN_FILE}" \
  --validation_file "${DATA_DIR}/${VALID_FILE}" \
  --classifier_configs "${CONFIG_PATH}" \
  --encoding_type bi_encoder \
  --max_seq_length ${MAX_SEQ_LEN} \
  --per_device_train_batch_size ${TRAIN_BATCH_SIZE} \
  --per_device_eval_batch_size ${EVAL_BATCH_SIZE} \
  --gradient_accumulation_steps ${GRAD_ACCUM} \
  --learning_rate ${LEARNING_RATE} \
  --num_train_epochs ${NUM_EPOCHS} \
  --lr_scheduler_type constant \
  --logging_steps 50 \
  --eval_strategy epoch \
  --save_strategy steps \
  --save_steps 5000 \
  --do_train \
  --seed ${SEED} \
  --fp16 ${FP16} \
  --bf16 ${BF16} \
  --freeze_encoder ${FREEZE_ENCODER} \
  --overwrite_output_dir True \
  --report_to none > "${OUTPUT_DIR}/${EXPERIMENT_NAME}/train.log" 2>&1
