#!/bin/bash
set -euo pipefail

# Joint-K4 on runtime participant-packed30 (DAIC): two seed-1337 backbone
# trainings (audio_only, audio_text), each dependency-chained
# 4-GPU train -> 1-GPU official-test eval (fp32) -> selected-epoch/full-cover
# hidden extraction -> logreg_raw/xgb_raw heads -> artifact audit.
# DRY_RUN=1 (default) only prints the plan. A real submission requires explicit
# user approval and DRY_RUN=0 plus a unique RUN_ID. Run this from an MN5
# scheduler login node (alogin2/alogin1); never transfer1. Submission is not
# completion; monitor every job to a terminal state with sacct.

PROJECT_ROOT="${PROJECT_ROOT:-/gpfs/projects/etur92/ozu647717/AudioLLM/LLM-Depression}"
ENV_ACTIVATE="${ENV_ACTIVATE:-/gpfs/projects/etur92/ozu647717/venvs/qwen_mn5_rebuilt/bin/activate}"
DRY_RUN="${DRY_RUN:-1}"
RUN_ID="${RUN_ID:-}"
SEED="${SEED:-1337}"
# Which stages to chain. Seed-variance search runs need only train+eval;
# the full chain (extract/heads/audit) is for finalists.
STAGES="${STAGES:-train eval extract heads audit}"
# Which modalities to submit (space-separated); search waves often run one.
MODALITIES="${MODALITIES:-audio_only audio_text}"
# Per-modality config overrides (variant search configs). Falls back to the
# base joint-K4 config for each modality when unset.
CONFIG_AUDIO_ONLY="${CONFIG_AUDIO_ONLY:-}"
CONFIG_AUDIO_TEXT="${CONFIG_AUDIO_TEXT:-}"
# Extra --set overrides passed to the train/eval workers (Tier-1 knobs).
EXTRA_TRAIN_ARGS="${EXTRA_TRAIN_ARGS:-}"
EXTRA_EVAL_ARGS="${EXTRA_EVAL_ARGS:-}"
EXPERIMENT_CONTEXT="${EXPERIMENT_CONTEXT:-}"
# Optional suffix appended to the generated run name (e.g. _ep40lr1e4) so
# multiple knob variants of the same seed/modality get distinct run dirs.
RUN_NAME_SUFFIX="${RUN_NAME_SUFFIX:-}"
CONFIG_DIR="$PROJECT_ROOT/configs/experiments/daic_participant_packed30_jointk4"
AUDIT_SCRIPT="$PROJECT_ROOT/scripts/audit_daic_participant_packed30_jointk4.py"
TRAIN_WORKER="$PROJECT_ROOT/scripts/run_daic_participant_packed30_jointk4_train_slurm.sh"
EVAL_WORKER="$PROJECT_ROOT/scripts/run_daic_participant_packed30_jointk4_eval_slurm.sh"
EXTRACT_WORKER="$PROJECT_ROOT/scripts/run_daic_participant_packed30_jointk4_extract_slurm.sh"
HEADS_WORKER="$PROJECT_ROOT/scripts/run_daic_participant_packed30_jointk4_heads_slurm.sh"
AUDIT_WORKER="$PROJECT_ROOT/scripts/run_daic_participant_packed30_jointk4_audit_slurm.sh"
DATASET_BASE_ROOT="${DATASET_BASE_ROOT:-/gpfs/projects/etur92/ozu647717/AudioLLM/Datasets}"
DAIC_UNPROCESSED_ROOT="${DAIC_UNPROCESSED_ROOT:-$DATASET_BASE_ROOT/DAIC-WOZ/unprocessed}"
DAIC_LABEL_ROOT="${DAIC_LABEL_ROOT:-$DATASET_BASE_ROOT/DAIC-WOZ/minimal_zips}"
MODEL_PATH="${MODEL_PATH:-/gpfs/projects/etur92/ozu647717/models/Qwen2-Audio-7B-Instruct}"
SUBMISSION_LOG="${SUBMISSION_LOG:-$PROJECT_ROOT/outputs/daic_participant_packed30_jointk4_jobs/$RUN_ID.tsv}"

if [ "$DRY_RUN" != "0" ] && [ "$DRY_RUN" != "1" ]; then
    echo "DRY_RUN must be 0 or 1." >&2
    exit 1
fi
if [ -z "$RUN_ID" ]; then
    echo "RUN_ID is required and must be unique (e.g. daic_participant_p30_jointk4_YYYYMMDD_<shortcommit>)." >&2
    exit 1
fi

SOURCE_COMMIT="$(cat "$PROJECT_ROOT/.provenance/git_commit.txt" 2>/dev/null || git -C "$PROJECT_ROOT" rev-parse HEAD 2>/dev/null || echo unknown)"
SHORTCOMMIT="$(printf '%s' "$SOURCE_COMMIT" | cut -c1-8)"

declare -A CONFIG_BY_MODALITY=(
    [audio_only]="${CONFIG_AUDIO_ONLY:-$CONFIG_DIR/daic_participant_packed30_jointk4_audio_only.yaml}"
    [audio_text]="${CONFIG_AUDIO_TEXT:-$CONFIG_DIR/daic_participant_packed30_jointk4_audio_text.yaml}"
)

for path in "$AUDIT_SCRIPT" "$TRAIN_WORKER" "$EVAL_WORKER" "$EXTRACT_WORKER" "$HEADS_WORKER" "$AUDIT_WORKER" "${CONFIG_BY_MODALITY[@]}"; do
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

export PROJECT_ROOT ENV_ACTIVATE DAIC_UNPROCESSED_ROOT DAIC_LABEL_ROOT SEED
export MODEL_PATH
EXPERIMENT_RUN_ROOT="$PROJECT_ROOT/output_model/experiments/daic_participant_packed30_jointk4"

declare -A CONFIG_BY_MODALITY
declare -A RUN_NAME_BY_MODALITY
declare -A FOLD_DIR_BY_MODALITY
declare -A MODALITY_JOBS

# Two-pass submission: stage jobs for the selected modalities are submitted
# before the next stage, so each audit waits on all chains' heads and never
# audits a half-finished run-root.
for modality in $MODALITIES; do
    config="${CONFIG_BY_MODALITY[$modality]}"
    run_name="daic_participant_p30_jointk4_${modality}_s${SEED}_${SHORTCOMMIT}${RUN_NAME_SUFFIX}"
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
        eval) worker="$EVAL_WORKER" ;;
        extract) worker="$EXTRACT_WORKER" ;;
        heads) worker="$HEADS_WORKER" ;;
        audit) worker="$AUDIT_WORKER" ;;
    esac
    local export_spec="PROJECT_ROOT=$PROJECT_ROOT,ENV_ACTIVATE=$ENV_ACTIVATE,CONFIG=$config,FOLD=0,RUN_NAME=$run_name,SEED=$SEED,DAIC_UNPROCESSED_ROOT=$DAIC_UNPROCESSED_ROOT,DAIC_LABEL_ROOT=$DAIC_LABEL_ROOT,MODEL_PATH=$MODEL_PATH,EXTRA_TRAIN_ARGS=$EXTRA_TRAIN_ARGS,EXTRA_EVAL_ARGS=$EXTRA_EVAL_ARGS"
    local ctx_var="EXPERIMENT_CONTEXT_${modality^^}"
    local modality_ctx="${!ctx_var:-$EXPERIMENT_CONTEXT}"
    if [ -n "$modality_ctx" ]; then export_spec="$export_spec,EXPERIMENT_CONTEXT=$modality_ctx"; fi
    if [ -n "$extra_export" ]; then export_spec="$export_spec,$extra_export"; fi
    local raw=""
    if [ "$DRY_RUN" = "1" ]; then
        echo "DRY RUN modality=$modality run_name=$run_name"
        echo "  $stage : sbatch ${dependency:+--dependency=$dependency }$worker"
        MODALITY_JOBS[$modality,${stage}_jobs]=$(( ${MODALITY_JOBS[$modality,${stage}_jobs]:-0} + 1 ))
        return
    fi
    local sbatch_cmd=(sbatch --parsable --job-name="jk4-${modality:0:6}-${stage:0:4}")
    if [ -n "$dependency" ]; then sbatch_cmd+=(--dependency="$dependency"); fi
    sbatch_cmd+=(--export="$export_spec" "$worker")
    raw="$("${sbatch_cmd[@]}")"
    local job_id="${raw%%;*}"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$stage" "$job_id" "$modality" "$run_name" "$dependency" "$config" "$SOURCE_COMMIT" >> "$SUBMISSION_LOG"
    echo "Submitted $stage modality=$modality job_id=$job_id (${dependency:-no dependency})"
    MODALITY_JOBS[$modality,$stage]="$job_id"
}

# Stage 1: both trains (independent, parallel).
for modality in $MODALITIES; do
    submit_stage train "$modality" "" ""
done
if [ "$DRY_RUN" = "1" ]; then
    for modality in $MODALITIES; do
        if [ "${MODALITY_JOBS[$modality,train_jobs]}" != "1" ]; then
            echo "DRY RUN expected exactly 1 backbone training for $modality; found a mismatch." >&2
            exit 1
        fi
    done
    echo "DRY RUN complete: no jobs submitted. Review the plan, then submit with DRY_RUN=0 and explicit approval."
    exit 0
fi
# Stage 2: evals after their own train.
for modality in $MODALITIES; do
    submit_stage eval "$modality" "afterok:${MODALITY_JOBS[$modality,train]}" ""
done
case " $STAGES " in
    *" extract "*) : ;;
    *) echo "STAGES=$STAGES: skipping extract/heads/audit (seed-variance mode)."; exit 0 ;;
esac
# Stage 3: both extracts after their own eval (fp32).
submit_stage extract audio_only "afterok:${MODALITY_JOBS[audio_only,eval]}" "CHECKPOINT_DIR=${FOLD_DIR_BY_MODALITY[audio_only]}/best_model,CACHE_DIR=$EXPERIMENT_RUN_ROOT/audio_only/${RUN_NAME_BY_MODALITY[audio_only]}/hidden_features/audio_only,CONDITION=audio_only"
submit_stage extract audio_text "afterok:${MODALITY_JOBS[audio_text,eval]}" "CHECKPOINT_DIR=${FOLD_DIR_BY_MODALITY[audio_text]}/best_model,CACHE_DIR=$EXPERIMENT_RUN_ROOT/audio_text/${RUN_NAME_BY_MODALITY[audio_text]}/hidden_features/audio_text,CONDITION=audio_text"
# Stage 4: both heads after their own extract.
submit_stage heads audio_only "afterok:${MODALITY_JOBS[audio_only,extract]}" "CACHE_DIR=$EXPERIMENT_RUN_ROOT/audio_only/${RUN_NAME_BY_MODALITY[audio_only]}/hidden_features/audio_only,OUTPUT_DIR=$EXPERIMENT_RUN_ROOT/audio_only/${RUN_NAME_BY_MODALITY[audio_only]}/hidden_classifiers/audio_only"
submit_stage heads audio_text "afterok:${MODALITY_JOBS[audio_text,extract]}" "CACHE_DIR=$EXPERIMENT_RUN_ROOT/audio_text/${RUN_NAME_BY_MODALITY[audio_text]}/hidden_features/audio_text,OUTPUT_DIR=$EXPERIMENT_RUN_ROOT/audio_text/${RUN_NAME_BY_MODALITY[audio_text]}/hidden_classifiers/audio_text"
# Stage 5: both audits, each waiting on BOTH chains' heads so the shared
# run-root is complete.
audit_extra="RUN_ROOT=$EXPERIMENT_RUN_ROOT,MANIFEST_DIR=$PROJECT_ROOT/outputs/manifests_daic_participant_packed30,SPLIT_DIR=$PROJECT_ROOT/outputs/splits_daic_participant_packed30,SMOKE=0"
submit_stage audit audio_only "afterok:${MODALITY_JOBS[audio_only,heads]},afterok:${MODALITY_JOBS[audio_text,heads]}" "$audit_extra"
submit_stage audit audio_text "afterok:${MODALITY_JOBS[audio_only,heads]},afterok:${MODALITY_JOBS[audio_text,heads]}" "$audit_extra"

echo "jointk4 production plan: train jobs=2 per-stage chains=2 run_id=$RUN_ID source_commit=$SOURCE_COMMIT"
echo "Submitted real jobs. Registry: $SUBMISSION_LOG"
echo "Submission is not completion: monitor squeue/sacct, sync results back (output_model minus best_model/last_model, logs/, outputs/), never rsync --delete, and run:"
echo "  python $AUDIT_SCRIPT --run-root $EXPERIMENT_RUN_ROOT --manifest-dir outputs/manifests_daic_participant_packed30 --split-dir outputs/splits_daic_participant_packed30"
