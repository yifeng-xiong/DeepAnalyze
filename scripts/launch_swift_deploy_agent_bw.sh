#!/usr/bin/env bash
# Launch the Agent SFT checkpoint as an OpenAI-compatible service via ms-swift.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SWIFT_ROOT="${SWIFT_ROOT:-$REPO_ROOT/deepanalyze/ms-swift}"
ENV_NAME="${ENV_NAME:-deepanalyze-bw-fast}"

if [[ "${CONDA_DEFAULT_ENV:-}" != "${ENV_NAME}" ]]; then
    CONDA_BASE="$(conda info --base 2>/dev/null || true)"
    if [[ -z "${CONDA_BASE}" || ! -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]]; then
        echo "Unable to locate conda.sh; please activate ${ENV_NAME} manually." >&2
        exit 1
    fi
    # Some activate hooks in this environment read unset variables.
    set +u
    # shellcheck source=/dev/null
    source "${CONDA_BASE}/etc/profile.d/conda.sh"
    conda activate "${ENV_NAME}"
    set -u
fi

export PYTHONNOUSERSITE=1
export DS_SKIP_CUDA_CHECK="${DS_SKIP_CUDA_CHECK:-1}"

SCRATCH="${SCRATCH:-/srv/disk00/yifengx4}"
export TMPDIR="${TMPDIR:-${SCRATCH}/tmp}"
LOCAL_TMP_ROOT="${LOCAL_TMP_ROOT:-${SCRATCH}/deepanalyze}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${LOCAL_TMP_ROOT}/triton-cache}"
export HF_HOME="${HF_HOME:-${LOCAL_TMP_ROOT}/hf-home}"
export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-${LOCAL_TMP_ROOT}/modelscope-cache}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-${LOCAL_TMP_ROOT}/torch-extensions}"
export VLLM_USE_V1="${VLLM_USE_V1:-0}"
mkdir -p "${TMPDIR}" "${TRITON_CACHE_DIR}" "${HF_HOME}" "${MODELSCOPE_CACHE}" "${TORCH_EXTENSIONS_DIR}"

CHECKPOINT_DIR="${CHECKPOINT_DIR:-/user-data/yifengx4/sft_model/v0-20260423-211414/checkpoint-144}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-LocalLLM}"
AGENT_TEMPLATE="${AGENT_TEMPLATE:-react_en}"
INFER_BACKEND="${INFER_BACKEND:-vllm}"
ATTN_IMPL="${ATTN_IMPL:-flash_attn}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"
DEPLOY_HOST="${DEPLOY_HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

if [[ ! -d "$SWIFT_ROOT" ]]; then
    echo "ms-swift not found: $SWIFT_ROOT" >&2
    exit 1
fi
if [[ ! -d "$CHECKPOINT_DIR" ]]; then
    echo "Checkpoint not found: $CHECKPOINT_DIR" >&2
    exit 1
fi

cd "$SWIFT_ROOT"

echo "Starting swift deploy service:"
echo "  model: ${CHECKPOINT_DIR}"
echo "  served_model_name: ${SERVED_MODEL_NAME}"
echo "  backend: ${INFER_BACKEND}"
echo "  agent_template: ${AGENT_TEMPLATE}"
echo "  attn_impl: ${ATTN_IMPL}"
echo "  listen: http://${DEPLOY_HOST}:${PORT}/v1"
echo "  CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "  VLLM_USE_V1: ${VLLM_USE_V1}"

swift deploy \
    --model "$CHECKPOINT_DIR" \
    --infer_backend "$INFER_BACKEND" \
    --attn_impl "$ATTN_IMPL" \
    --agent_template "$AGENT_TEMPLATE" \
    --served_model_name "$SERVED_MODEL_NAME" \
    --max_model_len "$MAX_MODEL_LEN" \
    --gpu_memory_utilization "$GPU_MEMORY_UTILIZATION" \
    --max_num_seqs "$MAX_NUM_SEQS" \
    --host "$DEPLOY_HOST" \
    --port "$PORT"
