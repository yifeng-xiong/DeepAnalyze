#!/usr/bin/env bash
# Full-parameter SFT on local Agent JSONL (tools + tool_call / tool_response), using ms-swift.
#
# Prereq: install ms-swift from vendored tree:
#   cd deepanalyze/ms-swift && pip install -e .
#
# Usage:
#   bash scripts/sft_swift_agent_full.sh
#   CUDA_VISIBLE_DEVICES=0 bash scripts/sft_swift_agent_full.sh
#
# On this repo's typical setup (conda + local HF/Triton caches), prefer:
#   bash scripts/run_sft_swift_agent_bw.sh
# (same idea as scripts/run_single_bw.sh → single.sh)
# Hugging Face Hub id instead of local dir:
#   MODEL=RUC-DataLab/DeepAnalyze-8B USE_HF=true bash scripts/sft_swift_agent_full.sh
#
# Agent template: try react_en; hermes matches many Qwen2.5 agent examples in swift.
# Loss on assistant spans: default --loss_scale react (override: LOSS_SCALE=default|hermes|...).
# ms-swift model group: default --model_type deepseek_r1_distill (override: MODEL_TYPE=qwen3 ...).
# Multi-GPU: set CUDA_VISIBLE_DEVICES=0,1,... and run with torchrun if your swift install supports it.
# ZeRO-3 + CPU optimizer offload (small-data full FT): DEEPSPEED_CONFIG=$ROOT/scripts/ds_zero3_optim_offload.json
#
# Note: full FT needs much more VRAM than LoRA; 8B + long agent traces often needs 40GB+ or ZeRO.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SWIFT_ROOT="${SWIFT_ROOT:-$ROOT/deepanalyze/ms-swift}"
DATASET="${DATASET:-$ROOT/data/swift_agent.jsonl}"
MODEL="${MODEL:-$ROOT/DeepAnalyze-8B}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/DeepAnalyze-8B-sft}"
# Local checkpoint: omit or USE_HF=false. Hub id (org/name): set USE_HF=true.
USE_HF="${USE_HF:-false}"

AGENT_TEMPLATE="${AGENT_TEMPLATE:-react_en}"
LOSS_SCALE="${LOSS_SCALE:-react}"
MODEL_TYPE="${MODEL_TYPE:-deepseek_r1_distill}"
TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-5}"
MAX_LENGTH="${MAX_LENGTH:-16384}"

LEARNING_RATE="${LEARNING_RATE:-1e-5}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-8}"
SPLIT_DATASET_RATIO="${SPLIT_DATASET_RATIO:-0.05}"
EVAL_STEPS="${EVAL_STEPS:-50}"
SAVE_STEPS="${SAVE_STEPS:-50}"
LOGGING_STEPS="${LOGGING_STEPS:-1}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-1}"
WARMUP_RATIO="${WARMUP_RATIO:-0.05}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-1}"
DATASET_NUM_PROC="${DATASET_NUM_PROC:-4}"
ATTN_IMPL="${ATTN_IMPL:-flash_attn}"
PACKING="${PACKING:-false}"
USE_LIGER_KERNEL="${USE_LIGER_KERNEL:-false}"
SAVE_ONLY_MODEL="${SAVE_ONLY_MODEL:-true}"

DEEPSPEED_ARGS=()
if [[ -n "${DEEPSPEED_CONFIG:-}" && "${DEEPSPEED_CONFIG}" != "none" ]]; then
  export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
  export MASTER_PORT="${MASTER_PORT:-29500}"
  export RANK="${RANK:-0}"
  export WORLD_SIZE="${WORLD_SIZE:-1}"
  export LOCAL_RANK="${LOCAL_RANK:-0}"
  DEEPSPEED_ARGS=(--deepspeed "${DEEPSPEED_CONFIG}")
fi

if [[ ! -f "$DATASET" ]]; then
  echo "Dataset not found: $DATASET" >&2
  exit 1
fi
if [[ ! -d "$SWIFT_ROOT" ]]; then
  echo "ms-swift dir not found: $SWIFT_ROOT" >&2
  exit 1
fi
if [[ ! -d "$MODEL" ]] && [[ "$USE_HF" != "true" ]]; then
  echo "Model path is not a directory: $MODEL" >&2
  echo "Set MODEL to your checkpoint dir, or use Hugging Face Hub: MODEL=org/name USE_HF=true" >&2
  exit 1
fi

cd "$SWIFT_ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

USE_HF_ARGS=()
if [[ "$USE_HF" == "true" ]]; then
  USE_HF_ARGS=(--use_hf true)
fi

swift sft \
  "${USE_HF_ARGS[@]}" \
  --model "$MODEL" \
  --model_type "$MODEL_TYPE" \
  --train_type full \
  --dataset "$DATASET" \
  --agent_template "$AGENT_TEMPLATE" \
  --loss_scale "$LOSS_SCALE" \
  --torch_dtype "$TORCH_DTYPE" \
  --num_train_epochs "$NUM_TRAIN_EPOCHS" \
  --per_device_train_batch_size "$PER_DEVICE_TRAIN_BATCH_SIZE" \
  --per_device_eval_batch_size 1 \
  --learning_rate "$LEARNING_RATE" \
  --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
  --split_dataset_ratio "$SPLIT_DATASET_RATIO" \
  --eval_steps "$EVAL_STEPS" \
  --save_steps "$SAVE_STEPS" \
  --save_total_limit "$SAVE_TOTAL_LIMIT" \
  --logging_steps "$LOGGING_STEPS" \
  --max_length "$MAX_LENGTH" \
  --output_dir "$OUTPUT_DIR" \
  --warmup_ratio "$WARMUP_RATIO" \
  --dataloader_num_workers "$DATALOADER_NUM_WORKERS" \
  --dataset_num_proc "$DATASET_NUM_PROC" \
  --attn_impl "$ATTN_IMPL" \
  --packing "$PACKING" \
  --use_liger_kernel "$USE_LIGER_KERNEL" \
  --lazy_tokenize true \
  --gradient_checkpointing true \
  --save_only_model "$SAVE_ONLY_MODEL" \
  --response_prefix "" \
  "${DEEPSPEED_ARGS[@]}"

echo "Done. Checkpoints under: $OUTPUT_DIR"
