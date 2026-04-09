#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${ENV_NAME:-deepanalyze-bw-fast}"

if [[ "${CONDA_DEFAULT_ENV:-}" != "${ENV_NAME}" ]]; then
    CONDA_BASE="$(conda info --base 2>/dev/null || true)"
    if [[ -z "${CONDA_BASE}" || ! -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]]; then
        echo "Unable to locate conda.sh; please activate ${ENV_NAME} manually."
        exit 1
    fi
    # Load conda in this non-interactive shell so the wrapper can activate the target env itself.
    # shellcheck source=/dev/null
    source "${CONDA_BASE}/etc/profile.d/conda.sh"
    conda activate "${ENV_NAME}"
fi

export PYTHONNOUSERSITE=1
export DS_SKIP_CUDA_CHECK="${DS_SKIP_CUDA_CHECK:-1}"

# Scratch on local disk: short-lived temp (TMPDIR) vs longer-lived caches (LOCAL_TMP_ROOT).
SCRATCH="${SCRATCH:-/srv/disk00/yifengx4}"
export TMPDIR="${TMPDIR:-${SCRATCH}/tmp}"
LOCAL_TMP_ROOT="${LOCAL_TMP_ROOT:-${SCRATCH}/deepanalyze}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${LOCAL_TMP_ROOT}/triton-cache}"
export HF_HOME="${HF_HOME:-${LOCAL_TMP_ROOT}/hf-home}"
export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-${LOCAL_TMP_ROOT}/modelscope-cache}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-${LOCAL_TMP_ROOT}/torch-extensions}"
mkdir -p "${TMPDIR}" "${TRITON_CACHE_DIR}" "${HF_HOME}" "${MODELSCOPE_CACHE}" "${TORCH_EXTENSIONS_DIR}"

cd "${REPO_ROOT}/deepanalyze/ms-swift"
exec bash "${REPO_ROOT}/scripts/single.sh"
