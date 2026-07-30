#!/bin/bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/gpfs/projects/etur92/ozu647717/AudioLLM/LLM-Depression}"
ENV_ACTIVATE="${ENV_ACTIVATE:-/gpfs/projects/etur92/ozu647717/venvs/qwen_mn5_rebuilt/bin/activate}"
DRY_RUN="${DRY_RUN:-1}"
RUN_ID="${RUN_ID:-androids_interview_$(date -u +%Y%m%dT%H%M%SZ)}"
FOLDS="${FOLDS:-0 1 2 3 4}"
SMOKE_AUDIT_PATH="${SMOKE_AUDIT_PATH:-}"
CONFIG_DIR="$PROJECT_ROOT/configs/experiments/androids_interview"
WORKER="$PROJECT_ROOT/scripts/run_androids_interview_worker_slurm.sh"
SUMMARY="$PROJECT_ROOT/scripts/run_androids_interview_summary_slurm.sh"
AUDIT_WORKER="$PROJECT_ROOT/scripts/run_androids_interview_audit_slurm.sh"
DATASET_BASE_ROOT="${DATASET_BASE_ROOT:-/gpfs/projects/etur92/ozu647717/AudioLLM/Datasets}"
ANDROIDS_DATASET_ROOT="${ANDROIDS_DATASET_ROOT:-$DATASET_BASE_ROOT/Androids-Corpus/Androids-Corpus}"
ANDROIDS_INTERVIEW_FULL_TRANSCRIPTS="${ANDROIDS_INTERVIEW_FULL_TRANSCRIPTS:-$ANDROIDS_DATASET_ROOT/interview_transcripts_qwen3_asr_italian.jsonl}"
ANDROIDS_INTERVIEW_SEGMENT_TRANSCRIPTS="${ANDROIDS_INTERVIEW_SEGMENT_TRANSCRIPTS:-$ANDROIDS_DATASET_ROOT/interview_transcripts_qwen3_asr_italian_segments.jsonl}"
SUBMISSION_LOG="${SUBMISSION_LOG:-$PROJECT_ROOT/outputs/androids_interview_jobs/$RUN_ID.tsv}"
CONFIGS=(
    "$CONFIG_DIR/androids_interview_audio_only.yaml"
    "$CONFIG_DIR/androids_interview_audio_text_segment_aligned.yaml"
    "$CONFIG_DIR/androids_interview_audio_text_full_turn.yaml"
    "$CONFIG_DIR/androids_interview_text_only.yaml"
)

if [ "$DRY_RUN" != "0" ] && [ "$DRY_RUN" != "1" ]; then
    echo "DRY_RUN must be 0 or 1." >&2
    exit 1
fi
if [ "$DRY_RUN" = "0" ]; then
    if [ -z "$SMOKE_AUDIT_PATH" ] || [ ! -s "$SMOKE_AUDIT_PATH" ]; then
        echo "Production requires SMOKE_AUDIT_PATH pointing to a passed smoke audit." >&2
        exit 1
    fi
    if ! python - "$SMOKE_AUDIT_PATH" <<'PY'
import json, sys
payload = json.load(open(sys.argv[1], encoding="utf-8"))
raise SystemExit(0 if payload.get("status") == "passed" and payload.get("mode") == "smoke" else 1)
PY
    then
        echo "Smoke audit did not pass: $SMOKE_AUDIT_PATH" >&2
        exit 1
    fi
    mkdir -p "$(dirname "$SUBMISSION_LOG")"
    if [ -e "$SUBMISSION_LOG" ]; then
        echo "Refusing colliding submission registry: $SUBMISSION_LOG" >&2
        exit 1
    fi
    SOURCE_COMMIT="$(cat "$PROJECT_ROOT/.provenance/git_commit.txt" 2>/dev/null || git -C "$PROJECT_ROOT" rev-parse HEAD 2>/dev/null || echo unknown)"
    printf 'timestamp_utc\tjob_type\tjob_id\tconfig\tfold\trun_name\trun_root\tsource_commit\n' > "$SUBMISSION_LOG"
fi
for path in "$WORKER" "$SUMMARY" "$AUDIT_WORKER" "${CONFIGS[@]}"; do
    if [ ! -f "$path" ]; then
        echo "Required implementation file is missing: $path" >&2
        exit 1
    fi
done
for path in "$ANDROIDS_INTERVIEW_FULL_TRANSCRIPTS" "$ANDROIDS_INTERVIEW_SEGMENT_TRANSCRIPTS"; do
    if [ "$DRY_RUN" = "0" ] && [ ! -s "$path" ]; then
        echo "Required transcript artifact is missing or empty: $path" >&2
        exit 1
    fi
done

export PROJECT_ROOT ANDROIDS_DATASET_ROOT
export ANDROIDS_INTERVIEW_FULL_TRANSCRIPTS ANDROIDS_INTERVIEW_SEGMENT_TRANSCRIPTS
MANIFEST_CMD=(python "$PROJECT_ROOT/src/data/build_manifest.py" --config "${CONFIGS[0]}")
if [ "$DRY_RUN" = "1" ]; then
    printf 'DRY RUN manifest:'
    printf ' %q' "${MANIFEST_CMD[@]}"
    printf '\n'
else
    # shellcheck disable=SC1090
    source "$ENV_ACTIVATE"
    cd "$PROJECT_ROOT"
    "${MANIFEST_CMD[@]}"
fi

GPU_JOBS=0
SUMMARY_JOBS=0
SUMMARY_IDS=()
RUN_SPECS=()
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
    RUN_SPECS+=("$config_id=$run_root")
    previous=""
    for fold in $FOLDS; do
        fold_dir="$run_root/fold_$fold"
        metrics="$fold_dir/eval/best_checkpoint/metrics_original_teacher_forced.json"
        selection="$fold_dir/logs/selected_checkpoint_selection_metrics.json"
        subjects="$fold_dir/eval/best_checkpoint/predictions_subject_level.csv"
        samples="$fold_dir/eval/best_checkpoint/predictions_sample_level.csv"
        responses="$fold_dir/eval/best_checkpoint/predictions_response_level.csv"
        completed=0
        if [ -s "$metrics" ] && [ -s "$selection" ] && [ -s "$fold_dir/run_config.yaml" ] \
            && [ -s "$subjects" ] && [ -s "$samples" ]; then
            if [ "$config_id" = "text_only" ] || [ -s "$responses" ]; then
                completed=1
            fi
        fi
        if [ "$completed" = "1" ]; then
            echo "SKIP structurally complete fold config=$stem fold=$fold path=$fold_dir"
            continue
        fi
        if [ -d "$fold_dir" ]; then
            echo "Refusing to overwrite incomplete/colliding fold directory: $fold_dir" >&2
            exit 1
        fi
        GPU_JOBS=$((GPU_JOBS + 1))
        export_spec="ALL,PROJECT_ROOT=$PROJECT_ROOT,ENV_ACTIVATE=$ENV_ACTIVATE,CONFIG=$config,FOLD=$fold,RUN_NAME=$run_name,DATASET_BASE_ROOT=$DATASET_BASE_ROOT,ANDROIDS_DATASET_ROOT=$ANDROIDS_DATASET_ROOT,ANDROIDS_INTERVIEW_FULL_TRANSCRIPTS=$ANDROIDS_INTERVIEW_FULL_TRANSCRIPTS,ANDROIDS_INTERVIEW_SEGMENT_TRANSCRIPTS=$ANDROIDS_INTERVIEW_SEGMENT_TRANSCRIPTS"
        dependency=()
        if [ -n "$previous" ]; then
            dependency=(--dependency="afterok:$previous")
        fi
        if [ "$DRY_RUN" = "1" ]; then
            echo "DRY RUN GPU config=$stem fold=$fold dependency=${previous:-none} run=$run_name root=$run_root"
            previous="DRY_${config_id}_${fold}"
        else
            raw="$(sbatch --parsable --job-name="and-${config_id:0:15}-f${fold}" "${dependency[@]}" --export="$export_spec" "$WORKER")"
            previous="${raw%%;*}"
            echo "Submitted GPU config=$stem fold=$fold job_id=$previous run=$run_name"
            printf '%s\tgpu\t%s\t%s\t%s\t%s\t%s\t%s\n' \
                "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$previous" "$stem" "$fold" \
                "$run_name" "$run_root" "$SOURCE_COMMIT" >> "$SUBMISSION_LOG"
        fi
    done
    SUMMARY_JOBS=$((SUMMARY_JOBS + 1))
    if [ "$DRY_RUN" = "1" ]; then
        echo "DRY RUN summary config=$stem dependency=${previous:-none} run_root=$run_root"
    else
        dependency_args=()
        if [ -n "$previous" ]; then
            dependency_args=(--dependency="afterok:$previous")
        fi
        raw="$(sbatch --parsable --job-name="and-${config_id:0:15}-sum" "${dependency_args[@]}" --export="ALL,PROJECT_ROOT=$PROJECT_ROOT,ENV_ACTIVATE=$ENV_ACTIVATE,RUN_ROOT=$run_root" "$SUMMARY")"
        summary_id="${raw%%;*}"
        SUMMARY_IDS+=("$summary_id")
        printf '%s\tsummary\t%s\t%s\t-\t%s\t%s\t%s\n' \
            "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$summary_id" "$stem" \
            "$run_name" "$run_root" "$SOURCE_COMMIT" >> "$SUBMISSION_LOG"
        echo "Submitted summary config=$stem job_id=$summary_id"
    fi
done

audit_out="$PROJECT_ROOT/outputs/androids_interview_matrix/$RUN_ID/audit.json"
echo "ANDROIDS Interview matrix plan: GPU jobs=$GPU_JOBS summary jobs=$SUMMARY_JOBS final audits=1 run_id=$RUN_ID"
if [ "$DRY_RUN" = "1" ]; then
    if [ "$FOLDS" = "0 1 2 3 4" ] && [ "$GPU_JOBS" -ne 20 ]; then
        echo "Dry-run expected 20 GPU jobs; $GPU_JOBS were planned because complete folds may have been skipped." >&2
    fi
    echo "DRY RUN expected fresh production total=25 jobs audit_output=$audit_out"
else
    summary_dependency="$(IFS=:; printf '%s' "${SUMMARY_IDS[*]}")"
    run_specs_joined="$(IFS=';'; printf '%s' "${RUN_SPECS[*]}")"
    raw="$(sbatch --parsable --job-name=androids-matrix-audit --dependency="afterok:$summary_dependency" --export="ALL,PROJECT_ROOT=$PROJECT_ROOT,ENV_ACTIVATE=$ENV_ACTIVATE,RUN_SPECS=$run_specs_joined,AUDIT_OUT=$audit_out,AUDIT_MODE=matrix" "$AUDIT_WORKER")"
    audit_id="${raw%%;*}"
    printf '%s\tmatrix_audit\t%s\tall\t-\t%s\t%s\t%s\n' \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$audit_id" "$RUN_ID" \
        "$audit_out" "$SOURCE_COMMIT" >> "$SUBMISSION_LOG"
    echo "Submitted final matrix audit job_id=$audit_id registry=$SUBMISSION_LOG"
fi
