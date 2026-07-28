#!/bin/bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/gpfs/projects/etur92/ozu647717/AudioLLM/LLM-Depression}"
ENV_ACTIVATE="${ENV_ACTIVATE:-/gpfs/projects/etur92/ozu647717/venvs/qwen_mn5_rebuilt/bin/activate}"
DRY_RUN="${DRY_RUN:-1}"
RUN_ID="${RUN_ID:-d3tec_$(date -u +%Y%m%dT%H%M%SZ)}"
FOLDS="${FOLDS:-0 1 2 3 4}"
CONFIG_DIR="$PROJECT_ROOT/configs/experiments/d3tec"
WORKER="$PROJECT_ROOT/scripts/run_d3tec_worker_slurm.sh"
SUMMARY="$PROJECT_ROOT/scripts/run_d3tec_summary_slurm.sh"
MATRIX_AUDIT="$PROJECT_ROOT/scripts/run_d3tec_matrix_audit_slurm.sh"
DATASET_BASE_ROOT="${DATASET_BASE_ROOT:-/gpfs/projects/etur92/ozu647717/AudioLLM/Datasets}"
D3TEC_DATASET_ROOT="${D3TEC_DATASET_ROOT:-$DATASET_BASE_ROOT/D3TEC DATASET/D3TEC DATASET}"
D3TEC_FULL_TRANSCRIPTS="${D3TEC_FULL_TRANSCRIPTS:-$D3TEC_DATASET_ROOT/transcripts_qwen3_asr_spanish.jsonl}"
D3TEC_SEGMENT_TRANSCRIPTS="${D3TEC_SEGMENT_TRANSCRIPTS:-$D3TEC_DATASET_ROOT/transcripts_qwen3_asr_spanish_segments.jsonl}"
SMOKE_AUDIT_PATH="${SMOKE_AUDIT_PATH:-}"
SUBMISSION_LOG="${SUBMISSION_LOG:-$PROJECT_ROOT/outputs/d3tec_jobs/$RUN_ID.tsv}"

CONFIGS=(
    "$CONFIG_DIR/d3tec_audio_only_rotary.yaml"
    "$CONFIG_DIR/d3tec_audio_only_flat.yaml"
    "$CONFIG_DIR/d3tec_audio_only_normalized.yaml"
    "$CONFIG_DIR/d3tec_audio_text_rotary.yaml"
    "$CONFIG_DIR/d3tec_audio_text_flat.yaml"
    "$CONFIG_DIR/d3tec_audio_text_normalized.yaml"
    "$CONFIG_DIR/d3tec_text_only.yaml"
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
raise SystemExit(0 if json.load(open(sys.argv[1], encoding="utf-8")).get("status") == "passed" else 1)
PY
    then
        echo "Smoke audit did not pass: $SMOKE_AUDIT_PATH" >&2
        exit 1
    fi
    mkdir -p "$(dirname "$SUBMISSION_LOG")"
    SOURCE_COMMIT="$(cat "$PROJECT_ROOT/.provenance/git_commit.txt" 2>/dev/null || git -C "$PROJECT_ROOT" rev-parse HEAD 2>/dev/null || echo unknown)"
    printf 'timestamp_utc\tjob_type\tjob_id\tconfig\tfold\trun_name\trun_root\tsource_commit\n' > "$SUBMISSION_LOG"
fi
for path in "$WORKER" "$SUMMARY" "$MATRIX_AUDIT" "${CONFIGS[@]}"; do
    if [ ! -f "$path" ]; then
        echo "Required file is missing: $path" >&2
        exit 1
    fi
done
for path in "$D3TEC_FULL_TRANSCRIPTS" "$D3TEC_SEGMENT_TRANSCRIPTS"; do
    if [ "$DRY_RUN" = "0" ] && [ ! -s "$path" ]; then
        echo "Required transcript artifact is missing or empty: $path" >&2
        exit 1
    fi
done

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

GPU_JOBS=0
SUMMARY_JOBS=0
SUMMARY_IDS=()
RUN_SPECS=()
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
    config_id="${stem#d3tec_}"
    RUN_SPECS+=("$config_id=$run_root")
    previous=""
    for fold in $FOLDS; do
        fold_dir="$run_root/fold_$fold"
        metrics="$fold_dir/eval/best_checkpoint/metrics_original_teacher_forced.json"
        selection="$fold_dir/logs/selected_checkpoint_selection_metrics.json"
        subject_predictions="$fold_dir/eval/best_checkpoint/predictions_subject_level.csv"
        sample_predictions="$fold_dir/eval/best_checkpoint/predictions_sample_level.csv"
        response_predictions="$fold_dir/eval/best_checkpoint/predictions_response_level.csv"
        completed=0
        if [ -s "$metrics" ] && [ -s "$selection" ] && [ -s "$fold_dir/run_config.yaml" ] \
            && [ -s "$subject_predictions" ] && [ -s "$sample_predictions" ]; then
            if [ "$stem" = "d3tec_text_only" ] || [ -s "$response_predictions" ]; then
                completed=1
            fi
        fi
        if [ "$completed" = "1" ]; then
            echo "SKIP audited existing fold config=$stem fold=$fold path=$fold_dir"
            continue
        fi
        if [ -d "$fold_dir" ]; then
            echo "Refusing to overwrite incomplete/colliding fold directory: $fold_dir" >&2
            exit 1
        fi
        GPU_JOBS=$((GPU_JOBS + 1))
        export_spec="ALL,PROJECT_ROOT=$PROJECT_ROOT,ENV_ACTIVATE=$ENV_ACTIVATE,CONFIG=$config,FOLD=$fold,RUN_NAME=$run_name,DATASET_BASE_ROOT=$DATASET_BASE_ROOT,D3TEC_DATASET_ROOT=$D3TEC_DATASET_ROOT,D3TEC_FULL_TRANSCRIPTS=$D3TEC_FULL_TRANSCRIPTS,D3TEC_SEGMENT_TRANSCRIPTS=$D3TEC_SEGMENT_TRANSCRIPTS"
        dependency=()
        if [ -n "$previous" ]; then
            dependency=(--dependency="afterok:$previous")
        fi
        if [ "$DRY_RUN" = "1" ]; then
            echo "DRY RUN GPU config=$stem fold=$fold dependency=${previous:-none} run=$run_name"
            previous="DRY_${stem}_${fold}"
        else
            raw="$(sbatch --parsable --job-name="d3-${stem:6:18}-f${fold}" "${dependency[@]}" --export="$export_spec" "$WORKER")"
            previous="${raw%%;*}"
            echo "Submitted GPU job config=$stem fold=$fold job_id=$previous run=$run_name"
            printf '%s\tgpu\t%s\t%s\t%s\t%s\t%s\t%s\n' \
                "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$previous" "$stem" "$fold" \
                "$run_name" "$run_root" "$SOURCE_COMMIT" >> "$SUBMISSION_LOG"
        fi
    done
    SUMMARY_JOBS=$((SUMMARY_JOBS + 1))
    if [ "$DRY_RUN" = "1" ]; then
        echo "DRY RUN summary config=$stem dependency=${previous:-none} run_root=$run_root"
    else
        summary_dependency_args=()
        if [ -n "$previous" ]; then
            summary_dependency_args=(--dependency="afterok:$previous")
        fi
        summary_raw="$(sbatch --parsable --job-name="d3-${stem:6:18}-sum" "${summary_dependency_args[@]}" --export="ALL,PROJECT_ROOT=$PROJECT_ROOT,ENV_ACTIVATE=$ENV_ACTIVATE,RUN_ROOT=$run_root" "$SUMMARY")"
        summary_id="${summary_raw%%;*}"
        SUMMARY_IDS+=("$summary_id")
        echo "Submitted summary config=$stem job_id=$summary_id"
        printf '%s\tsummary\t%s\t%s\t-\t%s\t%s\t%s\n' \
            "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$summary_id" "$stem" \
            "$run_name" "$run_root" "$SOURCE_COMMIT" >> "$SUBMISSION_LOG"
    fi
done

echo "D3TEC matrix plan: GPU jobs=$GPU_JOBS summary jobs=$SUMMARY_JOBS run_id=$RUN_ID"
if [ "$DRY_RUN" = "1" ] && [ "$FOLDS" = "0 1 2 3 4" ] && [ "$GPU_JOBS" -ne 35 ]; then
    echo "Dry-run expected 35 GPU jobs but planned $GPU_JOBS (completed folds may have been skipped)." >&2
fi
if [ "$DRY_RUN" = "1" ]; then
    echo "DRY RUN final matrix audit dependencies=7 summaries out=$PROJECT_ROOT/outputs/d3tec_matrix/$RUN_ID"
else
    summary_dependency="$(IFS=:; printf '%s' "${SUMMARY_IDS[*]}")"
    run_specs_joined="$(IFS=';'; printf '%s' "${RUN_SPECS[*]}")"
    audit_raw="$(sbatch --parsable --job-name=d3tec-matrix-audit --dependency="afterok:$summary_dependency" --export="ALL,PROJECT_ROOT=$PROJECT_ROOT,ENV_ACTIVATE=$ENV_ACTIVATE,RUN_SPECS=$run_specs_joined,AUDIT_OUT=$PROJECT_ROOT/outputs/d3tec_matrix/$RUN_ID" "$MATRIX_AUDIT")"
    audit_id="${audit_raw%%;*}"
    echo "Submitted final matrix audit job_id=$audit_id submission_log=$SUBMISSION_LOG"
    printf '%s\tmatrix_audit\t%s\tall\t-\t%s\t%s\t%s\n' \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$audit_id" "$RUN_ID" \
        "$PROJECT_ROOT/outputs/d3tec_matrix/$RUN_ID" "$SOURCE_COMMIT" >> "$SUBMISSION_LOG"
fi
