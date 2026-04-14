#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

mkdir -p condor/logs

CPUS="${CPUS:-4}"
MEMORY="${MEMORY:-16GB}"
FLAVOUR="${FLAVOUR:-nextweek}"

echo "Submitting Condor matrix with CPUS=${CPUS} MEMORY=${MEMORY} FLAVOUR=${FLAVOUR}"
condor_submit \
  -append "CPUS=${CPUS}" \
  -append "MEMORY=${MEMORY}" \
  -append "FLAVOUR=${FLAVOUR}" \
  condor/train_gpu.sub
