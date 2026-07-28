#!/bin/bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/gpfs/projects/etur92/ozu647717/AudioLLM/LLM-Depression}"
ENV_ACTIVATE="${ENV_ACTIVATE:-/gpfs/projects/etur92/ozu647717/venvs/qwen_mn5_rebuilt/bin/activate}"
DRY_RUN="${DRY_RUN:-1}"
RUN_ID="${RUN_ID:-d3tec_smoke_$(date -u +%Y%m%dT%H%M%SZ)}"
SMOKE_SUBJECT_LIMIT="${SMOKE_SUBJECT_LIMIT:-8}"
CONFIG_DIR="$PROJECT_ROOT/configs/experiments/d3tec"
WORKER="$PROJECT_ROOT/scripts/run_d3tec_worker_slurm.sh"
AUDIT_WORKER="$PROJECT_ROOT/scripts/run_d3tec_smoke_audit_slurm.sh"
DATASET_BASE_ROOT="${DATASET_BASE_ROOT:-/gpfs/projects/etur92/ozu647717/AudioLLM/Datasets}"
D3TEC_DATASET_ROOT="${D3TEC_DATASET_ROOT:-$DATASET_BASE_ROOT/D3TEC DATASET/D3TEC DATASET}"
D3TEC_FULL_TRANSCRIPTS="${D3TEC_FULL_TRANSCRIPTS:-$D3TEC_DATASET_ROOT/transcripts_qwen3_asr_spanish.jsonl}"
D3TEC_SEGMENT_TRANSCRIPTS="${D3TEC_SEGMENT_TRANSCRIPTS:-$D3TEC_DATASET_ROOT/transcripts_qwen3_asr_spanish_segments.jsonl}"
CONFIGS=(
    "$CONFIG_DIR/d3tec_audio_only_rotary.yaml"
    "$CONFIG_DIR/d3tec_audio_text_normalized.yaml"
    "$CONFIG_DIR/d3tec_text_only.yaml"
)
JOB_IDS=()
RUN_SPECS=()

if [ "$DRY_RUN" = "0" ]; then
    for path in "$D3TEC_FULL_TRANSCRIPTS" "$D3TEC_SEGMENT_TRANSCRIPTS"; do
        if [ ! -s "$path" ]; then
            echo "Required transcript artifact is missing: $path" >&2
            exit 1
        fi
    done
fi

export PROJECT_ROOT D3TEC_DATASET_ROOT D3TEC_FULL_TRANSCRIPTS D3TEC_SEGMENT_TRANSCRIPTS
MANIFEST_CMD=(python "$PROJECT_ROOT/src/data/build_manifest.py" --config "${CONFIGS[0]}")
if [ "$DRY_RUN" = "1" ]; then
    printf 'DRY RUN manifest:'
    printf ' %q' "${MANIFEST_CMD[@]}"
    printf '\n'
else
    # shellcheck disable=SC1090
    source "$ENV_ACTIVATE"
    "${MANIFEST_CMD[@]}"
fi

for config in "${CONFIGS[@]}"; do
    stem="$(basename "$config" .yaml)"
    run_name="${RUN_ID}_${stem}"
    run_root="$(PROJECT_ROOT="$PROJECT_ROOT" python - "$config" "$run_name" <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
from src.utils import load_yaml_with_overrides
config = load_yaml_with_overrides(sys.argv[1], [])
print(Path(config["output_dirs"]["run_root"]) / sys.argv[2])
PY
)"
    if [ -d "$run_root/fold_0" ]; then
        echo "Refusing to overwrite colliding smoke directory: $run_root/fold_0" >&2
        exit 1
    fi
    RUN_SPECS+=("${stem#d3tec_}=$run_root")
    if [ "$stem" = "d3tec_text_only" ]; then
        overrides="--set training.num_train_epochs=2 --set split.smoke_subject_limit=$SMOKE_SUBJECT_LIMIT"
    else
        overrides="--set training.num_train_epochs=2 --set training.reference_virtual_epochs=2 --set split.smoke_subject_limit=$SMOKE_SUBJECT_LIMIT"
    fi
    if [ "$DRY_RUN" = "1" ]; then
        echo "DRY RUN smoke config=$stem fold=0 run=$run_name path=$run_root overrides=$overrides"
        continue
    fi
    export_spec="ALL,PROJECT_ROOT=$PROJECT_ROOT,ENV_ACTIVATE=$ENV_ACTIVATE,CONFIG=$config,FOLD=0,RUN_NAME=$run_name,EXTRA_TRAIN_ARGS=$overrides,DATASET_BASE_ROOT=$DATASET_BASE_ROOT,D3TEC_DATASET_ROOT=$D3TEC_DATASET_ROOT,D3TEC_FULL_TRANSCRIPTS=$D3TEC_FULL_TRANSCRIPTS,D3TEC_SEGMENT_TRANSCRIPTS=$D3TEC_SEGMENT_TRANSCRIPTS"
    raw="$(sbatch --parsable --job-name="d3-smoke-${stem:6:18}" --export="$export_spec" "$WORKER")"
    job_id="${raw%%;*}"
    JOB_IDS+=("$job_id")
    echo "Submitted smoke config=$stem job_id=$job_id run=$run_name"
done
if [ "$DRY_RUN" = "1" ]; then
    echo "DRY RUN smoke audit dependencies=3 out=$PROJECT_ROOT/outputs/d3tec_smoke/$RUN_ID/audit.json"
else
    dependency="$(IFS=:; printf '%s' "${JOB_IDS[*]}")"
    run_specs_joined="$(IFS=';'; printf '%s' "${RUN_SPECS[*]}")"
    audit_out="$PROJECT_ROOT/outputs/d3tec_smoke/$RUN_ID/audit.json"
    raw="$(sbatch --parsable --dependency="afterok:$dependency" --export="ALL,PROJECT_ROOT=$PROJECT_ROOT,ENV_ACTIVATE=$ENV_ACTIVATE,RUN_SPECS=$run_specs_joined,AUDIT_OUT=$audit_out" "$AUDIT_WORKER")"
    echo "Submitted smoke audit job_id=${raw%%;*} output=$audit_out"
fi
