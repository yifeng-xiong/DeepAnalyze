export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
# Single-GPU: do not use torch.distributed.run (it SIGSEGV'd here with NPROC_PER_NODE=1).
# Clear any inherited NPROC_PER_NODE; for multi-GPU use a different entrypoint than single.sh.
unset NPROC_PER_NODE
if [[ -z "${MASTER_PORT:-}" ]]; then
    MASTER_PORT="$(
        python - <<'PY'
import socket

with socket.socket() as sock:
    sock.bind(("127.0.0.1", 0))
    print(sock.getsockname()[1])
PY
    )"
fi
export MASTER_PORT

BASE_MODEL="${BASE_MODEL:-/home/yifengx4/DeepAnalyze/models/DeepSeek-R1-0528-Qwen3-8B}"
MODEL_SINGLE_ABILITY_PATH="${MODEL_SINGLE_ABILITY_PATH:-/home/yifengx4/DeepAnalyze/models/DeepSeek-R1-0528-Qwen3-8B-sft}"
DATA_DIR="${DATA_DIR:-/home/yifengx4/DeepAnalyze/data/DataScience-Instruct-500K}"
TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-3}"
ATTN_IMPL="${ATTN_IMPL:-sdpa}"
# ms-swift requires flash_attn when packing is enabled; default false so sdpa works without flash-attn.
PACKING="${PACKING:-false}"
USE_LIGER_KERNEL="${USE_LIGER_KERNEL:-false}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
PER_DEVICE_EVAL_BATCH_SIZE="${PER_DEVICE_EVAL_BATCH_SIZE:-1}"
LEARNING_RATE="${LEARNING_RATE:-5e-5}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-32}"
EVAL_STEPS="${EVAL_STEPS:-50}"
SAVE_STEPS="${SAVE_STEPS:-50}"
LOGGING_STEPS="${LOGGING_STEPS:-1}"
MAX_LENGTH="${MAX_LENGTH:-8192}"
WARMUP_RATIO="${WARMUP_RATIO:-0.05}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-2}"
DATASET_NUM_PROC="${DATASET_NUM_PROC:-4}"
# Built-in names: zero0, zero1, zero2, zero3, zero2_offload, zero3_offload.
# Set to "none" to disable DeepSpeed (helps avoid crashes on some CUDA/driver stacks).
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-none}"

if [[ -n "${DATASET_OVERRIDE_FILE:-}" ]]; then
    DATASET_ARGS=(
        --dataset
        "${DATASET_OVERRIDE_FILE}"
    )
else
    DATASET_ARGS=(
        --dataset
        "${DATA_DIR}/reasoning/SKGInstruct_199989.json"
        "${DATA_DIR}/reasoning/TableQA_distillation_39301.json"
        "${DATA_DIR}/reasoning/TableQA_refinement_39301.json"
        "${DATA_DIR}/reasoning/TableGPT_29448.json"
        "${DATA_DIR}/reasoning/file_database_3833.json"
        "${DATA_DIR}/reasoning/file_csv_3007.json"
        "${DATA_DIR}/reasoning/file_xlsx_3663.json"
        "${DATA_DIR}/reasoning/file_any_2520.json"
        "${DATA_DIR}/reasoning/math_20000.json"
        "${DATA_DIR}/reasoning/code_20000.json"
        "${DATA_DIR}/reasoning/science_20000.json"
        "${DATA_DIR}/reasoning/instruction_following_20000.json"
        "${DATA_DIR}/reasoning/other_19998.json"
    )
fi

EXTRA_ARGS=()
if [[ -n "${MAX_STEPS:-}" ]]; then
    EXTRA_ARGS+=(--max_steps "${MAX_STEPS}")
fi

DEEPSPEED_ARGS=()
if [[ -n "${DEEPSPEED_CONFIG}" && "${DEEPSPEED_CONFIG}" != "none" ]]; then
    export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
    export RANK="${RANK:-0}"
    export WORLD_SIZE="${WORLD_SIZE:-1}"
    export LOCAL_RANK="${LOCAL_RANK:-0}"
    DEEPSPEED_ARGS=(--deepspeed "${DEEPSPEED_CONFIG}")
fi

# Make sure you are in directory ./deepanalyze/ms-swift/
swift sft \
    --model "${BASE_MODEL}" \
    --train_type "full" \
    "${DATASET_ARGS[@]}" \
    --torch_dtype "${TORCH_DTYPE}" \
    --num_train_epochs "${NUM_TRAIN_EPOCHS}" \
    --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}" \
    --per_device_eval_batch_size "${PER_DEVICE_EVAL_BATCH_SIZE}" \
    --learning_rate "${LEARNING_RATE}" \
    --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
    --packing "${PACKING}" \
    --eval_steps "${EVAL_STEPS}" \
    --save_steps "${SAVE_STEPS}" \
    --logging_steps "${LOGGING_STEPS}" \
    --max_length "${MAX_LENGTH}" \
    --warmup_ratio "${WARMUP_RATIO}" \
    --dataloader_num_workers "${DATALOADER_NUM_WORKERS}" \
    --dataset_num_proc "${DATASET_NUM_PROC}" \
    --lazy_tokenize true \
    --save_total_limit 1 \
    --response_prefix "" \
    --save_only_model false \
    --gradient_checkpointing true \
    --output_dir "${MODEL_SINGLE_ABILITY_PATH}" \
    "${DEEPSPEED_ARGS[@]}" \
    --use_liger_kernel "${USE_LIGER_KERNEL}" \
    --attn_impl "${ATTN_IMPL}" \
    --model_type "deepseek_r1_distill" \
    "${EXTRA_ARGS[@]}"