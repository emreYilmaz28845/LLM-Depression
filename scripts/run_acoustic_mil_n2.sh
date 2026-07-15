#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/emre/miniconda3/envs/llmdep4090/bin/python}"

cd "${ROOT}"
exec "${PYTHON_BIN}" -m src.baselines.acoustic_mil all "$@"
