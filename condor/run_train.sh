#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 5 ]]; then
  echo "Usage: $0 <label> <model> <dataset> <config> <extra_args>" >&2
  exit 2
fi

LABEL="$1"
MODEL="$2"
DATASET="$3"
CONFIG="$4"
EXTRA_ARGS="$5"
if [[ "${EXTRA_ARGS}" == "-" ]]; then
  EXTRA_ARGS=""
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export REPO_ROOT

cd "${REPO_ROOT}"

source "${REPO_ROOT}/scripts/setup_lcg.sh"

DATA_ROOT="${DATA_ROOT:-/eos/home-y/yeo/4l/data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/eos/home-y/yeo/4l/image}"

if [[ ! -d "${DATA_ROOT}" ]]; then
  echo "ERROR: DATA_ROOT does not exist: ${DATA_ROOT}" >&2
  exit 3
fi

if [[ ! -d "${OUTPUT_ROOT}" ]]; then
  mkdir -p "${OUTPUT_ROOT}"
fi

if [[ "${CONFIG}" != "-" ]]; then
  if [[ ! -f "${CONFIG}" ]]; then
    echo "ERROR: config file not found: ${CONFIG}" >&2
    exit 4
  fi
  CONFIG_ARGS=(--config "${CONFIG}")
else
  CONFIG_ARGS=()
fi

RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)_${CLUSTER_ID:-nocluster}_${PROCESS_ID:-noproc}"
RUN_DIR="${OUTPUT_ROOT}/${LABEL}/${MODEL//\//__}/${DATASET}/${RUN_ID}"
mkdir -p "${RUN_DIR}"

echo "[run_train] host=$(hostname)"
echo "[run_train] pwd=$(pwd)"
echo "[run_train] run_dir=${RUN_DIR}"
echo "[run_train] label=${LABEL} model=${MODEL} dataset=${DATASET} config=${CONFIG}"

echo "[run_train] command: python train_zarr_weighted.py --site lxplus --dataset ${DATASET} --label ${LABEL} --model ${MODEL} --output-root ${OUTPUT_ROOT} --run-id ${RUN_ID} --data-root ${DATA_ROOT} ${CONFIG_ARGS[*]} ${EXTRA_ARGS}"

set -x
python train_zarr_weighted.py \
  --site lxplus \
  --dataset "${DATASET}" \
  --label "${LABEL}" \
  --model "${MODEL}" \
  --output-root "${OUTPUT_ROOT}" \
  --run-id "${RUN_ID}" \
  --data-root "${DATA_ROOT}" \
  "${CONFIG_ARGS[@]}" \
  ${EXTRA_ARGS:-} \
  2>&1 | tee "${RUN_DIR}/train.stdout_stderr.log"
set +x
