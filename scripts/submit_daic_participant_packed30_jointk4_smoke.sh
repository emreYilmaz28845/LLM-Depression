#!/bin/bash
set -euo pipefail

# Joint-K4 runtime packed30 SMOKE chains (MN5 only; no model-bearing smoke
# locally): two parallel dependency chains (audio_only, audio_text), each
# 1-GPU smoke train (1 epoch, split.smoke_subject_limit=6, bf16)
# -> deterministic evaluation pass 1 -> deterministic evaluation pass 2 and
# comparison -> selected-epoch hidden extraction (bf16) -> logreg_raw/xgb_raw
# heads -> smoke artifact audit. No scientific score threshold. After both
# chains pass, report and STOP; production requires separate explicit approval.
# DRY_RUN=1 (default) only prints the plan. Run from an MN5 scheduler login
# node (alogin2/alogin1); never transfer1; never train directly on a login node.

PROJECT_ROOT="${PROJECT_ROOT:-/gpfs/projects/etur92/ozu647717/AudioLLM/LLM-Depression}"
ENV_ACTIVATE="${ENV_ACTIVATE:-/gpfs/projects/etur92/ozu647717/venvs/qwen_mn5_rebuilt/bin/activate}"
DRY_RUN="${DRY_RUN:-1}"
RUN_ID="${RUN_ID:-}"
SEED="${SEED:-1337}"
SMOKE_SUBJECT_LIMIT="${SMOKE_SUBJECT_LIMIT:-6}"
CONFIG_DIR="$PROJECT_ROOT/configs/experiments/daic_participant_packed30_jointk4"
AUDIT_SCRIPT="$PROJECT_ROOT/scripts/audit_daic_participant_packed30_jointk4.py"
TRAIN_WORKER="$PROJECT_ROOT/scripts/run_daic_participant_packed30_jointk4_smoke_train_slurm.sh"
EVAL_DETERMINISM_WORKER="$PROJECT_ROOT/scripts/run_daic_participant_packed30_jointk4_eval_determinism_slurm.sh"
EXTRACT_WORKER="$PROJECT_ROOT/scripts/run_daic_participant_packed30_jointk4_extract_slurm.sh"
HEADS_WORKER="$PROJECT_ROOT/scripts/run_daic_participant_packed30_jointk4_heads_slurm.sh"
AUDIT_WORKER="$PROJECT_ROOT/scripts/run_daic_participant_packed30_jointk4_audit_slurm.sh"
DATASET_BASE_ROOT="${DATASET_BASE_ROOT:-/gpfs/projects/etur92/ozu647717/AudioLLM/Datasets}"
DAIC_UNPROCESSED_ROOT="${DAIC_UNPROCESSED_ROOT:-$DATASET_BASE_ROOT/DAIC-WOZ/unprocessed}"
DAIC_LABEL_ROOT="${DAIC_LABEL_ROOT:-$DATASET_BASE_ROOT/DAIC-WOZ/minimal_zips}"
MODEL_PATH="${MODEL_PATH:-/gpfs/projects/etur92/ozu647717/models/Qwen2-Audio-7B-Instruct}"
SUBMISSION_LOG="${SUBMISSION_LOG:-$PROJECT_ROOT/outputs/daic_participant_packed30_jointk4_jobs/smoke_$RUN_ID.tsv}"

if [ "$DRY_RUN" != "0" ] && [ "$DRY_RUN" != "1" ]; then
    echo "DRY_RUN must be 0 or 1." >&2
    exit 1
fi
if [ -z "$RUN_ID" ]; then
    echo "RUN_ID is required and must be unique (e.g. smoke_p30_jointk4_YYYYMMDD_<shortcommit>)." >&2
    exit 1
fi

SOURCE_COMMIT="$(cat "$PROJECT_ROOT/.provenance/git_commit.txt" 2>/dev/null || git -C "$PROJECT_ROOT" rev-parse HEAD 2>/dev/null || echo unknown)"
SHORTCOMMIT="$(printf '%s' "$SOURCE_COMMIT" | cut -c1-8)"

declare -A CONFIG_BY_MODALITY=(
    [audio_only]="$CONFIG_DIR/daic_participant_packed30_jointk4_audio_only.yaml"
    [audio_text]="$CONFIG_DIR/daic_participant_packed30_jointk4_audio_text.yaml"
)

for path in "$AUDIT_SCRIPT" "$TRAIN_WORKER" "$EVAL_DETERMINISM_WORKER" "$EXTRACT_WORKER" "$HEADS_WORKER" "$AUDIT_WORKER" "${CONFIG_BY_MODALITY[@]}"; do
    if [ ! -f "$path" ]; then
        echo "Required implementation file is missing: $path" >&2
        exit 1
    fi
done
if [ "$DRY_RUN" = "0" ]; then
    if [ ! -d "$DAIC_UNPROCESSED_ROOT" ] || [ ! -d "$DAIC_LABEL_ROOT" ]; then
        echo "DAIC corpus roots are missing on the cluster." >&2
        exit 1
    fi
    mkdir -p "$(dirname "$SUBMISSION_LOG")"
    if [ -e "$SUBMISSION_LOG" ]; then
        echo "Refusing colliding submission registry: $SUBMISSION_LOG" >&2
        exit 1
    fi
    printf 'timestamp_utc\tstage\tjob_id\tmodality\trun_name\tdependency\tconfig\tsource_commit\n' > "$SUBMISSION_LOG"
fi

export PROJECT_ROOT ENV_ACTIVATE DAIC_UNPROCESSED_ROOT DAIC_LABEL_ROOT SEED SMOKE_SUBJECT_LIMIT
export MODEL_PATH
EXPERIMENT_RUN_ROOT="$PROJECT_ROOT/output_model/experiments/daic_participant_packed30_jointk4"

declare -A CONFIG_BY_MODALITY
declare -A RUN_NAME_BY_MODALITY
declare -A FOLD_DIR_BY_MODALITY
declare -A MODALITY_JOBS

# Two-pass submission: stage jobs for BOTH modalities are submitted before the
# next stage, so each audit waits on BOTH chains' heads and never audits a
# half-finished run-root.
for modality in audio_only audio_text; do
    config="${CONFIG_BY_MODALITY[$modality]}"
    run_name="smoke_p30_jointk4_${modality}_s${SEED}_${SHORTCOMMIT}"
    run_root="$(python - "$config" "$run_name" <<PY
import sys
from pathlib import Path
sys.path.insert(0, "$PROJECT_ROOT")
from src.utils import load_yaml_with_overrides
config = load_yaml_with_overrides(sys.argv[1], [])
print(Path(config["output_dirs"]["run_root"]) / sys.argv[2])
PY
)"
    fold_dir="$run_root/fold_0"
    if [ -e "$fold_dir" ]; then
        echo "Refusing overwrite of existing run directory: $fold_dir" >&2
        exit 1
    fi
    CONFIG_BY_MODALITY[$modality]="$config"
    RUN_NAME_BY_MODALITY[$modality]="$run_name"
    FOLD_DIR_BY_MODALITY[$modality]="$fold_dir"
done

submit_stage() {
    local stage="$1" modality="$2" dependency="$3" extra_export="$4"
    local config="${CONFIG_BY_MODALITY[$modality]}"
    local run_name="${RUN_NAME_BY_MODALITY[$modality]}"
    local worker=""
    case "$stage" in
        train) worker="$TRAIN_WORKER" ;;
        eval_det) worker="$EVAL_DETERMINISM_WORKER" ;;
        extract) worker="$EXTRACT_WORKER" ;;
        heads) worker="$HEADS_WORKER" ;;
        audit) worker="$AUDIT_WORKER" ;;
    esac
    local export_spec="PROJECT_ROOT=$PROJECT_ROOT,ENV_ACTIVATE=$ENV_ACTIVATE,CONFIG=$config,FOLD=0,RUN_NAME=$run_name,SEED=$SEED,SMOKE_SUBJECT_LIMIT=$SMOKE_SUBJECT_LIMIT,DAIC_UNPROCESSED_ROOT=$DAIC_UNPROCESSED_ROOT,DAIC_LABEL_ROOT=$DAIC_LABEL_ROOT,MODEL_PATH=$MODEL_PATH"
    if [ -n "$extra_export" ]; then export_spec="$export_spec,$extra_export"; fi
    local raw=""
    if [ "$DRY_RUN" = "1" ]; then
        echo "DRY RUN modality=$modality run_name=$run_name"
        echo "  $stage : sbatch ${dependency:+--dependency=$dependency }$worker"
        MODALITY_JOBS[$modality,${stage}_jobs]=$(( ${MODALITY_JOBS[$modality,${stage}_jobs]:-0} + 1 ))
        return
    fi
    local sbatch_cmd=(sbatch --parsable --job-name="jk4-smoke-${modality:0:6}-${stage:0:4}")
    if [ -n "$dependency" ]; then sbatch_cmd+=(--dependency="$dependency"); fi
    sbatch_cmd+=(--export="$export_spec" "$worker")
    raw="$("${sbatch_cmd[@]}")"
    local job_id="${raw%%;*}"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$stage" "$job_id" "$modality" "$run_name" "$dependency" "$config" "$SOURCE_COMMIT" >> "$SUBMISSION_LOG"
    echo "Submitted $stage modality=$modality job_id=$job_id (${dependency:-no dependency})"
    MODALITY_JOBS[$modality,$stage]="$job_id"
}

# Stage 1: both smoke trains (independent, parallel; 1 GPU each).
submit_stage train audio_only "" ""
submit_stage train audio_text "" ""
if [ "$DRY_RUN" = "1" ]; then
    if [ "${MODALITY_JOBS[audio_only,train_jobs]}" != "1" ] || [ "${MODALITY_JOBS[audio_text,train_jobs]}" != "1" ]; then
        echo "DRY RUN expected exactly 2 backbone trainings; found a mismatch." >&2
        exit 1
    fi
    echo "DRY RUN complete: no jobs submitted. Review the plan, then submit with DRY_RUN=0 and explicit approval."
    exit 0
fi
# Stage 2: both determinism evals after their own train (bf16, pass1+pass2+compare).
submit_stage eval_det audio_only "afterok:${MODALITY_JOBS[audio_only,train]}" ""
submit_stage eval_det audio_text "afterok:${MODALITY_JOBS[audio_text,train]}" ""
# Stage 3: both selected-epoch extracts after their own eval (bf16).
submit_stage extract audio_only "afterok:${MODALITY_JOBS[audio_only,eval_det]}" "CHECKPOINT_DIR=${FOLD_DIR_BY_MODALITY[audio_only]}/best_model,CACHE_DIR=$EXPERIMENT_RUN_ROOT/audio_only/${RUN_NAME_BY_MODALITY[audio_only]}/hidden_features/audio_only,CONDITION=audio_only,EXTRACTION_INFERENCE_DTYPE=bf16"
submit_stage extract audio_text "afterok:${MODALITY_JOBS[audio_text,eval_det]}" "CHECKPOINT_DIR=${FOLD_DIR_BY_MODALITY[audio_text]}/best_model,CACHE_DIR=$EXPERIMENT_RUN_ROOT/audio_text/${RUN_NAME_BY_MODALITY[audio_text]}/hidden_features/audio_text,CONDITION=audio_text,EXTRACTION_INFERENCE_DTYPE=bf16"
# Stage 4: both heads after their own extract.
submit_stage heads audio_only "afterok:${MODALITY_JOBS[audio_only,extract]}" "CACHE_DIR=$EXPERIMENT_RUN_ROOT/audio_only/${RUN_NAME_BY_MODALITY[audio_only]}/hidden_features/audio_only,OUTPUT_DIR=$EXPERIMENT_RUN_ROOT/audio_only/${RUN_NAME_BY_MODALITY[audio_only]}/hidden_classifiers/audio_only"
submit_stage heads audio_text "afterok:${MODALITY_JOBS[audio_text,extract]}" "CACHE_DIR=$EXPERIMENT_RUN_ROOT/audio_text/${RUN_NAME_BY_MODALITY[audio_text]}/hidden_features/audio_text,OUTPUT_DIR=$EXPERIMENT_RUN_ROOT/audio_text/${RUN_NAME_BY_MODALITY[audio_text]}/hidden_classifiers/audio_text"
# Stage 5: both smoke audits, each waiting on BOTH chains' heads.
audit_extra="RUN_ROOT=$EXPERIMENT_RUN_ROOT,MANIFEST_DIR=$PROJECT_ROOT/outputs/manifests_daic_participant_packed30,SPLIT_DIR=$PROJECT_ROOT/outputs/splits_daic_participant_packed30,SMOKE=1"
submit_stage audit audio_only "afterok:${MODALITY_JOBS[audio_only,heads]},afterok:${MODALITY_JOBS[audio_text,heads]}" "$audit_extra"
submit_stage audit audio_text "afterok:${MODALITY_JOBS[audio_only,heads]},afterok:${MODALITY_JOBS[audio_text,heads]}" "$audit_extra"

echo "jointk4 smoke plan: train jobs=2 per-stage chains=2 run_id=$RUN_ID source_commit=$SOURCE_COMMIT"
echo "Submitted real smoke jobs. Registry: $SUBMISSION_LOG"
echo "Submission is not completion: monitor squeue/sacct, verify all jobs COMPLETED exit 0:0, then STOP."
echo "Production requires a separate explicit user confirmation."
