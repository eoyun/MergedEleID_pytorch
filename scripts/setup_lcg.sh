#!/usr/bin/env bash
set -euo pipefail

# Bootstrap an LCG Python environment on lxplus/HTCondor workers.
# Usage:
#   source scripts/setup_lcg.sh

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "ERROR: source this script, do not execute it directly: source scripts/setup_lcg.sh" >&2
  exit 1
fi

LCG_BASE="/cvmfs/sft.cern.ch/lcg/views"

if [[ ! -d "${LCG_BASE}" ]]; then
  echo "ERROR: LCG base path not found: ${LCG_BASE}" >&2
  return 1
fi

if [[ -n "${LCG_VIEW:-}" ]]; then
  CANDIDATES=("${LCG_VIEW}")
else
  CANDIDATES=(
    "${LCG_BASE}/LCG_106/x86_64-el9-gcc13-opt/setup.sh"
    "${LCG_BASE}/LCG_106/x86_64-el9-gcc12-opt/setup.sh"
    "${LCG_BASE}/LCG_105a/x86_64-el9-gcc12-opt/setup.sh"
    "${LCG_BASE}/LCG_105/x86_64-centos7-gcc11-opt/setup.sh"
  )
fi

CHOSEN=""
for candidate in "${CANDIDATES[@]}"; do
  if [[ -f "${candidate}" ]]; then
    CHOSEN="${candidate}"
    break
  fi
done

if [[ -z "${CHOSEN}" ]]; then
  echo "ERROR: no usable LCG view found." >&2
  echo "Tried:" >&2
  printf '  - %s\n' "${CANDIDATES[@]}" >&2
  echo "Set LCG_VIEW to an explicit setup.sh path and retry." >&2
  return 1
fi

# shellcheck disable=SC1090
source "${CHOSEN}"
export LCG_VIEW_RESOLVED="${CHOSEN}"

echo "[setup_lcg] Using LCG view: ${LCG_VIEW_RESOLVED}"
echo "[setup_lcg] Python: $(command -v python)"
python --version

# Optional thin venv layer for pip-only deps; default off for pure-LCG mode.
if [[ "${LCG_LAYERED_VENV:-0}" == "1" ]]; then
  if [[ -z "${REPO_ROOT:-}" ]]; then
    echo "ERROR: REPO_ROOT must be set when LCG_LAYERED_VENV=1" >&2
    return 1
  fi
  REQ_FILE="${LCG_REQUIREMENTS_FILE:-${REPO_ROOT}/requirements.txt}"
  if [[ ! -f "${REQ_FILE}" ]]; then
    echo "ERROR: requirements file for layered venv not found: ${REQ_FILE}" >&2
    return 1
  fi

  CACHE_BASE="${LCG_VENV_CACHE:-${HOME}/.cache/mergedeleid_lcg_venvs}"
  mkdir -p "${CACHE_BASE}"
  REQ_HASH="$(sha256sum "${REQ_FILE}" | awk '{print $1}')"
  PY_TAG="$(python - <<'PY'
import sys
print(f"py{sys.version_info.major}{sys.version_info.minor}")
PY
)"
  VENV_PATH="${CACHE_BASE}/${PY_TAG}_${REQ_HASH:0:12}"

  if [[ ! -x "${VENV_PATH}/bin/python" ]]; then
    echo "[setup_lcg] Creating layered venv at ${VENV_PATH}"
    python -m venv "${VENV_PATH}"
    "${VENV_PATH}/bin/pip" install --upgrade pip
    "${VENV_PATH}/bin/pip" install -r "${REQ_FILE}"
  else
    echo "[setup_lcg] Reusing layered venv at ${VENV_PATH}"
  fi

  # shellcheck disable=SC1090
  source "${VENV_PATH}/bin/activate"
fi
