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
TRAIN_JOBS=0
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

    export CONFIG="$config" RUN_NAME="$run_name" FOLD=0
    export MODEL_PATH
    eval_export_spec="PROJECT_ROOT=$PROJECT_ROOT,ENV_ACTIVATE=$ENV_ACTIVATE,CONFIG=$config,FOLD=0,RUN_NAME=$run_name,SEED=$SEED,SMOKE_SUBJECT_LIMIT=$SMOKE_SUBJECT_LIMIT,DAIC_UNPROCESSED_ROOT=$DAIC_UNPROCESSED_ROOT,DAIC_LABEL_ROOT=$DAIC_LABEL_ROOT,MODEL_PATH=$MODEL_PATH"

    if [ "$DRY_RUN" = "1" ]; then
        TRAIN_JOBS=$((TRAIN_JOBS + 1))
        echo "DRY RUN modality=$modality run_name=$run_name"
        echo "  smoke-train : sbatch --gres=gpu:1 --time=24:00:00 (1 GPU, 1 epoch, $SMOKE_SUBJECT_LIMIT subjects) $TRAIN_WORKER"
        echo "  eval det    : --dependency=afterok:<train> --gres=gpu:1 --time=24:00:00 (bf16, pass1+pass2+compare) $EVAL_DETERMINISM_WORKER"
        echo "  extract     : --dependency=afterok:<eval-det> --gres=gpu:1 --time=24:00:00 (bf16) $EXTRACT_WORKER -> hidden_features/$modality"
        echo "  heads       : --dependency=afterok:<extract> (0 GPUs, 4 CPUs) $HEADS_WORKER -> logreg_raw + xgb_raw"
        echo "  audit       : --dependency=afterok:<heads> (0 GPUs, 4 CPUs, --smoke) $AUDIT_WORKER"
        continue
    fi

    train_raw="$(sbatch --parsable --job-name="jk4-smoke-${modality:0:6}" \
        --export="$eval_export_spec" "$TRAIN_WORKER")"
    train_id="${train_raw%%;*}"
    TRAIN_JOBS=$((TRAIN_JOBS + 1))
    printf '%s\tsmoke_train\t%s\t%s\t%s\t-\t%s\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$train_id" "$modality" "$run_name" "$config" "$SOURCE_COMMIT" >> "$SUBMISSION_LOG"
    echo "Submitted smoke train modality=$modality job_id=$train_id run_name=$run_name"

    eval_raw="$(sbatch --parsable --job-name="jk4-smoke-${modality:0:6}-eval" \
        --dependency="afterok:$train_id" --export="$eval_export_spec" "$EVAL_DETERMINISM_WORKER")"
    eval_id="${eval_raw%%;*}"
    printf '%s\teval_determinism\t%s\t%s\t%s\t%s\t%s\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$eval_id" "$modality" "$run_name" "$train_id" "$config" "$SOURCE_COMMIT" >> "$SUBMISSION_LOG"
    echo "Submitted eval determinism modality=$modality job_id=$eval_id (afterok:$train_id)"

    cache_dir="$run_root/hidden_features/$modality"
    extract_export_spec="$eval_export_spec,CHECKPOINT_DIR=$fold_dir/best_model,CACHE_DIR=$cache_dir,CONDITION=$modality,EXTRACTION_INFERENCE_DTYPE=bf16"
    extract_raw="$(sbatch --parsable --job-name="jk4-smoke-${modality:0:6}-extract" \
        --dependency="afterok:$eval_id" --export="$extract_export_spec" "$EXTRACT_WORKER")"
    extract_id="${extract_raw%%;*}"
    printf '%s\textract\t%s\t%s\t%s\t%s\t%s\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$extract_id" "$modality" "$run_name" "$eval_id" "$config" "$SOURCE_COMMIT" >> "$SUBMISSION_LOG"
    echo "Submitted extract modality=$modality job_id=$extract_id (afterok:$eval_id)"

    heads_dir="$run_root/hidden_classifiers/$modality"
    heads_export_spec="$eval_export_spec,CACHE_DIR=$cache_dir,OUTPUT_DIR=$heads_dir"
    heads_raw="$(sbatch --parsable --job-name="jk4-smoke-${modality:0:6}-heads" \
        --dependency="afterok:$extract_id" --export="$heads_export_spec" "$HEADS_WORKER")"
    heads_id="${heads_raw%%;*}"
    printf '%s\theads\t%s\t%s\t%s\t%s\t%s\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$heads_id" "$modality" "$run_name" "$extract_id" "$config" "$SOURCE_COMMIT" >> "$SUBMISSION_LOG"
    echo "Submitted heads modality=$modality job_id=$heads_id (afterok:$extract_id)"

    audit_export_spec="$eval_export_spec,RUN_ROOT=$run_root,MANIFEST_DIR=$PROJECT_ROOT/outputs/manifests_daic_participant_packed30,SPLIT_DIR=$PROJECT_ROOT/outputs/splits_daic_participant_packed30,SMOKE=1"
    audit_raw="$(sbatch --parsable --job-name="jk4-smoke-${modality:0:6}-audit" \
        --dependency="afterok:$heads_id" --export="$audit_export_spec" "$AUDIT_WORKER")"
    audit_id="${audit_raw%%;*}"
    printf '%s\taudit\t%s\t%s\t%s\t%s\t%s\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$audit_id" "$modality" "$run_name" "$heads_id" "$config" "$SOURCE_COMMIT" >> "$SUBMISSION_LOG"
    echo "Submitted audit modality=$modality job_id=$audit_id (afterok:$heads_id)"
done

echo "jointk4 smoke plan: train jobs=$TRAIN_JOBS per-stage chains=2 run_id=$RUN_ID source_commit=$SOURCE_COMMIT"
if [ "$DRY_RUN" = "1" ]; then
    if [ "$TRAIN_JOBS" -ne 2 ]; then
        echo "DRY RUN expected exactly 2 backbone trainings; found $TRAIN_JOBS." >&2
        exit 1
    fi
    echo "DRY RUN complete: no jobs submitted. Review the plan, then submit with DRY_RUN=0 and explicit approval."
else
    echo "Submitted real smoke jobs. Registry: $SUBMISSION_LOG"
    echo "Submission is not completion: monitor squeue/sacct, verify all jobs COMPLETED exit 0:0, then STOP."
    echo "Production requires a separate explicit user confirmation."
fi
