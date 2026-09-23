#!/bin/bash
#SBATCH -J qwen3omni-probe
#SBATCH -A etur92
#SBATCH -q acc_ehpc
#SBATCH -t 06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=80
#SBATCH --gres=gpu:4
#SBATCH -o /dev/null
#SBATCH -e /dev/null
#SBATCH --chdir=/gpfs/projects/etur92/ozu647717/AudioLLM/LLM-Depression

# Qwen3-Omni probe worker: one configurable shape, several probe modes.
#
#   PROBE_MODE=forward|backward|maxrisk|perf   (required)
#   CONFIG=<config path>                       (required)
#   PROBE_OUTPUT=<json path>                   (required)
#   PROBE_INVENTORY=<risk inventory json>      (for the risk-example selection)
#   PROBE_GPUS=4 PROBE_NODES=1                 (shape; sbatch overrides the header)
#   PROBE_ARGS="--steps 3 --modalities audio_only audio_text"
#   OVERRIDES_JSON_B64 / EXTRA_PROBE_ARGS: the lossless common override transport
#
# `forward` runs as one process with a device map across the requested GPUs; the
# FSDP modes run one rank per GPU through torchrun. The worker never trains a
# production run and never writes into a run directory.

set -e
set -o pipefail

module purge
module load bsc/1.0
module load miniforge/24.3.0-0

# The Qwen3-Omni environment is dedicated and offline: the Qwen2-Audio
# environment ships transformers 4.55.0 without the Qwen3OmniMoe classes.
ENV_ACTIVATE="${ENV_ACTIVATE:-/gpfs/projects/etur92/ozu647717/venvs/qwen3omni/bin/activate}"
if [ -f "$ENV_ACTIVATE" ]; then
    # shellcheck disable=SC1090
    source "$ENV_ACTIVATE"
else
    echo "Environment activate script not found: $ENV_ACTIVATE"
    exit 1
fi

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

PROJECT_ROOT="${PROJECT_ROOT:-/gpfs/projects/etur92/ozu647717/AudioLLM/LLM-Depression}"
export PROJECT_ROOT
cd "$PROJECT_ROOT"

DATASET_BASE_ROOT="${DATASET_BASE_ROOT:-/gpfs/projects/etur92/ozu647717/AudioLLM/Datasets}"
export DAIC_DATASET_ROOT="${DAIC_DATASET_ROOT:-$DATASET_BASE_ROOT/DAIC-WOZ/preprocessed}"
export DAIC_UNPROCESSED_ROOT="${DAIC_UNPROCESSED_ROOT:-$DATASET_BASE_ROOT/DAIC-WOZ/unprocessed}"
export DAIC_LABEL_ROOT="${DAIC_LABEL_ROOT:-$DATASET_BASE_ROOT/DAIC-WOZ/minimal_zips}"

PROBE_MODE="${PROBE_MODE:?Set PROBE_MODE (forward|backward|maxrisk|perf)}"
CONFIG="${CONFIG:?Set CONFIG}"
PROBE_OUTPUT="${PROBE_OUTPUT:?Set PROBE_OUTPUT}"
PROBE_INVENTORY="${PROBE_INVENTORY:-}"
PROBE_GPUS="${PROBE_GPUS:-4}"
PROBE_NODES="${PROBE_NODES:-1}"
PROBE_ARGS="${PROBE_ARGS:-}"
EXTRA_PROBE_ARGS="${EXTRA_PROBE_ARGS:-}"
OVERRIDES_JSON_B64="${OVERRIDES_JSON_B64:-}"
# Probe logs land beside the probe output (the task runtime), never inside the
# immutable source deployment: writing there would dirty a clean-source tree.
LOG_ROOT="${LOG_ROOT:-$(dirname "$PROBE_OUTPUT")/logs}"

# The interpreter of the activated environment drives the ranks (same reasoning
# as the training worker: torchrun's shebang would otherwise pick the base env).
if [ -n "${VIRTUAL_ENV:-}" ] && [ -x "$VIRTUAL_ENV/bin/python" ]; then
    PYTHON_BIN="$VIRTUAL_ENV/bin/python"
else
    PYTHON_BIN="$(command -v python)"
fi

# Lossless common overrides: the base64 JSON token array is authoritative when
# present; the whitespace-split string is only a legacy fallback.
if [ -n "$OVERRIDES_JSON_B64" ]; then
    mapfile -t -d '' OVERRIDE_ARGS < <("$PYTHON_BIN" - "$OVERRIDES_JSON_B64" <<'PY'
import base64, json, sys
tokens = json.loads(base64.b64decode(sys.argv[1]).decode("utf-8"))
sys.stdout.write("\0".join(tokens))
PY
)
else
    read -r -a OVERRIDE_ARGS <<< "$EXTRA_PROBE_ARGS"
fi
read -r -a PROBE_EXTRA <<< "$PROBE_ARGS"

mkdir -p "$LOG_ROOT"
TIMESTAMP="$(date +%Y-%m-%d_%H:%M:%S)"
RUN_LOG="$LOG_ROOT/${PROBE_MODE}-${SLURM_JOB_ID:-local}-${TIMESTAMP}.log"
exec > >(tee -a "$RUN_LOG") 2>&1

echo "========================================"
echo "Qwen3-Omni probe | mode=$PROBE_MODE | job=${SLURM_JOB_ID:-}"
echo "config=$CONFIG"
echo "output=$PROBE_OUTPUT"
echo "inventory=${PROBE_INVENTORY:-<none>}"
echo "shape=${PROBE_NODES} node(s) x ${PROBE_GPUS} GPU(s)"
echo "overrides=${OVERRIDE_ARGS[*]:-<none>}"
echo "probe_args=${PROBE_EXTRA[*]:-<none>}"
echo "host=$(hostname)"
echo "========================================"
nvidia-smi || true
"$PYTHON_BIN" -V
"$PYTHON_BIN" -c "import torch, transformers, peft, accelerate; print('torch', torch.__version__, 'cuda', torch.version.cuda); print('transformers', transformers.__version__); print('peft', peft.__version__); print('accelerate', accelerate.__version__)"

# torchrun runs its entrypoint with the interpreter that runs it, so the
# entrypoint is the script plus its arguments. Passing the venv interpreter as
# the first element would make torchrun execute that binary as a Python script.
PROBE_ENTRYPOINT=(
    scripts/qwen3omni_backend_probe.py "$PROBE_MODE"
    --config "$CONFIG"
    --output "$PROBE_OUTPUT"
    "${OVERRIDE_ARGS[@]}"
)
if [ -n "$PROBE_INVENTORY" ]; then
    PROBE_ENTRYPOINT+=(--inventory "$PROBE_INVENTORY")
fi
PROBE_ENTRYPOINT+=("${PROBE_EXTRA[@]}")

if [ "$PROBE_MODE" = "forward" ] || [ "$PROBE_MODE" = "tree" ]; then
    echo "launching single-process ${PROBE_MODE} probe"
    "$PYTHON_BIN" "${PROBE_ENTRYPOINT[@]}"
else
    echo "launching torchrun probe | nodes=$PROBE_NODES | ranks/node=$PROBE_GPUS"
    if [ "${PROBE_NODES}" -gt 1 ]; then
        MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
        MASTER_PORT="${MASTER_PORT:-29517}"
        srun --nodes="$PROBE_NODES" --ntasks="$PROBE_NODES" --ntasks-per-node=1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-80}" --export=ALL \
            bash -c 'exec "$0" -m torch.distributed.run --nproc_per_node="$1" --nnodes="$2" --node_rank="$SLURM_NODEID" --master_addr="$3" --master_port="$4" "${@:5}"' \
            "$PYTHON_BIN" "$PROBE_GPUS" "$PROBE_NODES" "$MASTER_ADDR" "$MASTER_PORT" "${PROBE_ENTRYPOINT[@]}"
    else
        "$PYTHON_BIN" -m torch.distributed.run --nproc_per_node="$PROBE_GPUS" --standalone "${PROBE_ENTRYPOINT[@]}"
    fi
fi

echo "probe complete: $PROBE_OUTPUT"
