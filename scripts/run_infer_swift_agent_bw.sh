#!/usr/bin/env bash
# Batch inference for Agent SFT checkpoints (ms-swift), same conda/cache setup as
# scripts/run_sft_swift_agent_bw.sh.
#
#   bash scripts/run_infer_swift_agent_bw.sh
#
# Override examples:
#   CHECKPOINT_DIR=/path/to/checkpoint-80 \
#   VAL_DATASET=/path/to/test.jsonl \
#   RESULT_PATH=/path/to/out.jsonl \
#   bash scripts/run_infer_swift_agent_bw.sh
#
#   CUDA_VISIBLE_DEVICES=1 bash scripts/run_infer_swift_agent_bw.sh
#
# Note: swift infer appends to result_path; delete the file first if you want a clean run.

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
    # shellcheck source=/dev/null
    source "${CONDA_BASE}/etc/profile.d/conda.sh"
    conda activate "${ENV_NAME}"
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
mkdir -p "${TMPDIR}" "${TRITON_CACHE_DIR}" "${HF_HOME}" "${MODELSCOPE_CACHE}" "${TORCH_EXTENSIONS_DIR}"

CHECKPOINT_DIR="${CHECKPOINT_DIR:-/srv/disk00/yifengx4/swift_agent_local_llm_sft/v7-20260408-170250/checkpoint-80}"
VAL_DATASET="${VAL_DATASET:-$REPO_ROOT/data/swift_agent_toolcall_infer_smoke.jsonl}"
RESULT_PATH="${RESULT_PATH:-/srv/disk00/yifengx4/swift_agent_local_llm_sft/v7-20260408-170250/infer_result.jsonl}"
AGENT_TEMPLATE="${AGENT_TEMPLATE:-react_en}"
INFER_BACKEND="${INFER_BACKEND:-pt}"
MAX_BATCH_SIZE="${MAX_BATCH_SIZE:-1}"
TEMPERATURE="${TEMPERATURE:-0}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

if [[ ! -d "$SWIFT_ROOT" ]]; then
    echo "ms-swift not found: $SWIFT_ROOT" >&2
    exit 1
fi
if [[ ! -d "$CHECKPOINT_DIR" ]]; then
    echo "Checkpoint not found: $CHECKPOINT_DIR" >&2
    exit 1
fi
if [[ ! -f "$VAL_DATASET" ]]; then
    echo "val_dataset not found: $VAL_DATASET" >&2
    exit 1
fi

if [[ "${TRUNCATE_RESULT:-0}" == "1" || "${TRUNCATE_RESULT:-}" == "true" ]]; then
    rm -f "$RESULT_PATH"
fi

cd "$SWIFT_ROOT"

swift infer \
    --model "$CHECKPOINT_DIR" \
    --val_dataset "$VAL_DATASET" \
    --agent_template "$AGENT_TEMPLATE" \
    --infer_backend "$INFER_BACKEND" \
    --max_batch_size "$MAX_BATCH_SIZE" \
    --temperature "$TEMPERATURE" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --result_path "$RESULT_PATH"

echo "Done. Results: $RESULT_PATH"
