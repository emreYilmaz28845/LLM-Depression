#!/bin/bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/gpfs/projects/etur92/ozu647717/AudioLLM/LLM-Depression}"
ENV_ACTIVATE="${ENV_ACTIVATE:-/gpfs/projects/etur92/ozu647717/venvs/qwen_mn5_rebuilt/bin/activate}"
DRY_RUN="${DRY_RUN:-1}"
RUN_ID="${RUN_ID:-androids_interview_smoke_$(date -u +%Y%m%dT%H%M%SZ)}"
SMOKE_SUBJECT_LIMIT="${SMOKE_SUBJECT_LIMIT:-24}"
CONFIG_DIR="$PROJECT_ROOT/configs/experiments/androids_interview"
WORKER="$PROJECT_ROOT/scripts/run_androids_interview_worker_slurm.sh"
AUDIT_WORKER="$PROJECT_ROOT/scripts/run_androids_interview_audit_slurm.sh"
DATASET_BASE_ROOT="${DATASET_BASE_ROOT:-/gpfs/projects/etur92/ozu647717/AudioLLM/Datasets}"
ANDROIDS_DATASET_ROOT="${ANDROIDS_DATASET_ROOT:-$DATASET_BASE_ROOT/Androids-Corpus/Androids-Corpus}"
ANDROIDS_INTERVIEW_FULL_TRANSCRIPTS="${ANDROIDS_INTERVIEW_FULL_TRANSCRIPTS:-$ANDROIDS_DATASET_ROOT/interview_transcripts_qwen3_asr_italian.jsonl}"
ANDROIDS_INTERVIEW_SEGMENT_TRANSCRIPTS="${ANDROIDS_INTERVIEW_SEGMENT_TRANSCRIPTS:-$ANDROIDS_DATASET_ROOT/interview_transcripts_qwen3_asr_italian_segments.jsonl}"
CONFIGS=(
    "$CONFIG_DIR/androids_interview_audio_only.yaml"
    "$CONFIG_DIR/androids_interview_audio_text_segment_aligned.yaml"
    "$CONFIG_DIR/androids_interview_audio_text_full_turn.yaml"
    "$CONFIG_DIR/androids_interview_text_only.yaml"
)
JOB_IDS=()
RUN_SPECS=()

if [ "$DRY_RUN" != "0" ] && [ "$DRY_RUN" != "1" ]; then
    echo "DRY_RUN must be 0 or 1." >&2
    exit 1
fi
if [ "$DRY_RUN" = "0" ]; then
    for path in "$ANDROIDS_INTERVIEW_FULL_TRANSCRIPTS" "$ANDROIDS_INTERVIEW_SEGMENT_TRANSCRIPTS"; do
        if [ ! -s "$path" ]; then
            echo "Required transcript artifact is missing: $path" >&2
            exit 1
        fi
    done
fi
for path in "$WORKER" "$AUDIT_WORKER" "${CONFIGS[@]}"; do
    if [ ! -f "$path" ]; then
        echo "Required implementation file is missing: $path" >&2
        exit 1
    fi
done

export PROJECT_ROOT ANDROIDS_DATASET_ROOT
export ANDROIDS_INTERVIEW_FULL_TRANSCRIPTS ANDROIDS_INTERVIEW_SEGMENT_TRANSCRIPTS
MANIFEST_CMD=(python "$PROJECT_ROOT/src/data/build_manifest.py" --config "${CONFIGS[0]}")
INPUT_AUDIT_OUT="$PROJECT_ROOT/outputs/androids_interview_smoke/$RUN_ID/input_audit.json"
INPUT_AUDIT_CMD=(
    python "$PROJECT_ROOT/scripts/audit_androids_interview_inputs.py"
    --dataset-root "$ANDROIDS_DATASET_ROOT"
    --full-transcripts "$ANDROIDS_INTERVIEW_FULL_TRANSCRIPTS"
    --segment-transcripts "$ANDROIDS_INTERVIEW_SEGMENT_TRANSCRIPTS"
    --out "$INPUT_AUDIT_OUT"
)
if [ "$DRY_RUN" = "1" ]; then
    printf 'DRY RUN manifest:'
    printf ' %q' "${MANIFEST_CMD[@]}"
    printf '\n'
    printf 'DRY RUN input audit:'
    printf ' %q' "${INPUT_AUDIT_CMD[@]}"
    printf '\n'
else
    # shellcheck disable=SC1090
    source "$ENV_ACTIVATE"
    cd "$PROJECT_ROOT"
    "${MANIFEST_CMD[@]}"
    "${INPUT_AUDIT_CMD[@]}"
fi

for config in "${CONFIGS[@]}"; do
    stem="$(basename "$config" .yaml)"
    config_id="${stem#androids_interview_}"
    run_name="${RUN_ID}_${stem}"
    run_root="$(cd "$PROJECT_ROOT" && python - "$config" "$run_name" <<'PY'
import sys
from pathlib import Path
from src.utils import load_yaml_with_overrides
config = load_yaml_with_overrides(sys.argv[1], [])
print(Path(config["output_dirs"]["run_root"]) / sys.argv[2])
PY
)"
    if [ -d "$run_root/fold_0" ]; then
        echo "Refusing to overwrite colliding smoke directory: $run_root/fold_0" >&2
        exit 1
    fi
    RUN_SPECS+=("$config_id=$run_root")
    overrides="--set training.num_train_epochs=2 --set split.smoke_subject_limit=$SMOKE_SUBJECT_LIMIT"
    if [ "$DRY_RUN" = "1" ]; then
        echo "DRY RUN smoke config=$stem fold=0 run=$run_name path=$run_root overrides=$overrides"
        continue
    fi
    export_spec="ALL,PROJECT_ROOT=$PROJECT_ROOT,ENV_ACTIVATE=$ENV_ACTIVATE,CONFIG=$config,FOLD=0,RUN_NAME=$run_name,EXTRA_TRAIN_ARGS=$overrides,DATASET_BASE_ROOT=$DATASET_BASE_ROOT,ANDROIDS_DATASET_ROOT=$ANDROIDS_DATASET_ROOT,ANDROIDS_INTERVIEW_FULL_TRANSCRIPTS=$ANDROIDS_INTERVIEW_FULL_TRANSCRIPTS,ANDROIDS_INTERVIEW_SEGMENT_TRANSCRIPTS=$ANDROIDS_INTERVIEW_SEGMENT_TRANSCRIPTS"
    raw="$(sbatch --parsable --job-name="and-smoke-${config_id:0:16}" --export="$export_spec" "$WORKER")"
    job_id="${raw%%;*}"
    JOB_IDS+=("$job_id")
    echo "Submitted smoke config=$stem job_id=$job_id run=$run_name"
done

audit_out="$PROJECT_ROOT/outputs/androids_interview_smoke/$RUN_ID/audit.json"
if [ "$DRY_RUN" = "1" ]; then
    echo "DRY RUN smoke plan: GPU jobs=4 audit jobs=1 output=$audit_out"
else
    dependency="$(IFS=:; printf '%s' "${JOB_IDS[*]}")"
    run_specs_joined="$(IFS=';'; printf '%s' "${RUN_SPECS[*]}")"
    raw="$(sbatch --parsable --dependency="afterok:$dependency" --export="ALL,PROJECT_ROOT=$PROJECT_ROOT,ENV_ACTIVATE=$ENV_ACTIVATE,RUN_SPECS=$run_specs_joined,AUDIT_OUT=$audit_out,AUDIT_MODE=smoke" "$AUDIT_WORKER")"
    echo "Submitted smoke audit job_id=${raw%%;*} output=$audit_out"
fi
