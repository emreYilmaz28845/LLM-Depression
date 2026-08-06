#!/bin/bash
set -euo pipefail

# v1 participant-packed30 SMOKE submission: one seed-1337 backbone training per
# modality (1 GPU, 1 epoch, split.smoke_subject_limit=6), dependency-chained
# train -> official-test eval -> hidden extraction -> logreg_raw/xgb_raw heads.
# DRY_RUN=1 (default) only prints the plan. Smoke runs never replace production;
# production submission uses scripts/submit_daic_participant_packed30.sh.

PROJECT_ROOT="${PROJECT_ROOT:-/gpfs/projects/etur92/ozu647717/AudioLLM/LLM-Depression}"
ENV_ACTIVATE="${ENV_ACTIVATE:-/gpfs/projects/etur92/ozu647717/venvs/qwen_mn5_rebuilt/bin/activate}"
DRY_RUN="${DRY_RUN:-1}"
RUN_ID="${RUN_ID:-}"
SEED="${SEED:-1337}"
SMOKE_SUBJECT_LIMIT="${SMOKE_SUBJECT_LIMIT:-6}"
CONFIG_DIR="$PROJECT_ROOT/configs/experiments/daic_participant_packed30"
AUDIT_SCRIPT="$PROJECT_ROOT/scripts/audit_daic_participant_packed30.py"
TRAIN_WORKER="$PROJECT_ROOT/scripts/run_daic_participant_packed30_smoke_train_slurm.sh"
EVAL_WORKER="$PROJECT_ROOT/scripts/run_daic_participant_packed30_eval_slurm.sh"
EXTRACT_WORKER="$PROJECT_ROOT/scripts/run_daic_participant_packed30_extract_slurm.sh"
HEADS_WORKER="$PROJECT_ROOT/scripts/run_daic_participant_packed30_heads_slurm.sh"
DATASET_BASE_ROOT="${DATASET_BASE_ROOT:-/gpfs/projects/etur92/ozu647717/AudioLLM/Datasets}"
DAIC_UNPROCESSED_ROOT="${DAIC_UNPROCESSED_ROOT:-$DATASET_BASE_ROOT/DAIC-WOZ/unprocessed}"
DAIC_LABEL_ROOT="${DAIC_LABEL_ROOT:-$DATASET_BASE_ROOT/DAIC-WOZ/minimal_zips}"
MODEL_PATH="${MODEL_PATH:-/gpfs/projects/etur92/ozu647717/models/Qwen2-Audio-7B-Instruct}"
TEXT_MODEL_PATH="${TEXT_MODEL_PATH:-/gpfs/projects/etur92/ozu647717/models/Qwen2-7B-Instruct}"
INFERENCE_DTYPE="${INFERENCE_DTYPE:-bf16}"
SUBMISSION_LOG="${SUBMISSION_LOG:-$PROJECT_ROOT/outputs/daic_participant_packed30_jobs/$RUN_ID.tsv}"

if [ "$DRY_RUN" != "0" ] && [ "$DRY_RUN" != "1" ]; then
    echo "DRY_RUN must be 0 or 1." >&2
    exit 1
fi
if [ -z "$RUN_ID" ]; then
    echo "RUN_ID is required and must be unique (e.g. daic_participant_p30_smoke_YYYYMMDD_<shortcommit>)." >&2
    exit 1
fi

SOURCE_COMMIT="$(cat "$PROJECT_ROOT/.provenance/git_commit.txt" 2>/dev/null || git -C "$PROJECT_ROOT" rev-parse HEAD 2>/dev/null || echo unknown)"
SHORTCOMMIT="$(printf '%s' "$SOURCE_COMMIT" | cut -c1-8)"

declare -A CONFIG_BY_MODALITY=(
    [audio_only]="$CONFIG_DIR/daic_participant_packed30_audio_only.yaml"
    [audio_text]="$CONFIG_DIR/daic_participant_packed30_audio_text.yaml"
    [text_only]="$CONFIG_DIR/daic_participant_full_transcript_text_only.yaml"
)

for path in "$AUDIT_SCRIPT" "$TRAIN_WORKER" "$EVAL_WORKER" "$EXTRACT_WORKER" "$HEADS_WORKER" "${CONFIG_BY_MODALITY[@]}"; do
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

export PROJECT_ROOT ENV_ACTIVATE DAIC_UNPROCESSED_ROOT DAIC_LABEL_ROOT SEED SMOKE_SUBJECT_LIMIT INFERENCE_DTYPE
TRAIN_JOBS=0
for modality in audio_only audio_text text_only; do
    config="${CONFIG_BY_MODALITY[$modality]}"
    run_name="smoke_p30_${modality}_s${SEED}_${SHORTCOMMIT}"
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
    # MODEL_PATH is read with precedence by resolve_model_name_or_path, so it
    # must only be exported for audio modalities; text-only must fall back to
    # the config default (TEXT_MODEL_PATH-resolved Qwen2-7B-Instruct).
    if [ "$modality" = "text_only" ]; then
        modality_model_spec="TEXT_MODEL_PATH=$TEXT_MODEL_PATH"
    else
        modality_model_spec="MODEL_PATH=$MODEL_PATH"
    fi
    stage_export_spec="ALL,PROJECT_ROOT=$PROJECT_ROOT,ENV_ACTIVATE=$ENV_ACTIVATE,CONFIG=$config,FOLD=0,RUN_NAME=$run_name,SEED=$SEED,SMOKE_SUBJECT_LIMIT=$SMOKE_SUBJECT_LIMIT,INFERENCE_DTYPE=$INFERENCE_DTYPE,DAIC_UNPROCESSED_ROOT=$DAIC_UNPROCESSED_ROOT,DAIC_LABEL_ROOT=$DAIC_LABEL_ROOT,$modality_model_spec"

    if [ "$DRY_RUN" = "1" ]; then
        TRAIN_JOBS=$((TRAIN_JOBS + 1))
        echo "DRY RUN modality=$modality run_name=$run_name"
        echo "  train   : sbatch --gres=gpu:1 --time=24:00:00 (1 GPU, 1 epoch, $SMOKE_SUBJECT_LIMIT subjects) $TRAIN_WORKER"
        echo "  eval    : --dependency=afterok:<train> --gres=gpu:1 $EVAL_WORKER -> $fold_dir/best_model/standalone_eval"
        echo "  extract : --dependency=afterok:<eval>  --gres=gpu:1 $EXTRACT_WORKER -> hidden cache"
        echo "  heads   : --dependency=afterok:<extract> (0 GPUs) $HEADS_WORKER -> logreg_raw + xgb_raw"
        continue
    fi

    train_raw="$(sbatch --parsable --job-name="p30sm-${modality:0:9}-tr" \
        --export="$stage_export_spec" "$TRAIN_WORKER")"
    train_id="${train_raw%%;*}"
    TRAIN_JOBS=$((TRAIN_JOBS + 1))
    printf '%s\tsmoke_train\t%s\t%s\t%s\t-\t%s\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$train_id" "$modality" "$run_name" "$config" "$SOURCE_COMMIT" >> "$SUBMISSION_LOG"
    echo "Submitted smoke train modality=$modality job_id=$train_id run_name=$run_name"

    eval_raw="$(sbatch --parsable --job-name="p30sm-${modality:0:9}-ev" \
        --dependency="afterok:$train_id" --export="$stage_export_spec" "$EVAL_WORKER")"
    eval_id="${eval_raw%%;*}"
    printf '%s\tsmoke_eval\t%s\t%s\t%s\t%s\t%s\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$eval_id" "$modality" "$run_name" "$train_id" "$config" "$SOURCE_COMMIT" >> "$SUBMISSION_LOG"
    echo "Submitted smoke eval modality=$modality job_id=$eval_id (afterok:$train_id)"

    cache_dir="$run_root/hidden_features/$modality"
    extract_export_spec="$stage_export_spec,CHECKPOINT_DIR=$fold_dir/best_model,CACHE_DIR=$cache_dir,CONDITION=$modality,EXTRACTION_INFERENCE_DTYPE=$INFERENCE_DTYPE"
    extract_raw="$(sbatch --parsable --job-name="p30sm-${modality:0:9}-ex" \
        --dependency="afterok:$eval_id" --export="$extract_export_spec" "$EXTRACT_WORKER")"
    extract_id="${extract_raw%%;*}"
    printf '%s\tsmoke_extract\t%s\t%s\t%s\t%s\t%s\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$extract_id" "$modality" "$run_name" "$eval_id" "$config" "$SOURCE_COMMIT" >> "$SUBMISSION_LOG"
    echo "Submitted smoke extract modality=$modality job_id=$extract_id (afterok:$eval_id)"

    heads_dir="$run_root/hidden_classifiers/$modality"
    heads_export_spec="$stage_export_spec,CACHE_DIR=$cache_dir,OUTPUT_DIR=$heads_dir"
    heads_raw="$(sbatch --parsable --job-name="p30sm-${modality:0:9}-hd" \
        --dependency="afterok:$extract_id" --export="$heads_export_spec" "$HEADS_WORKER")"
    heads_id="${heads_raw%%;*}"
    printf '%s\tsmoke_heads\t%s\t%s\t%s\t%s\t%s\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$heads_id" "$modality" "$run_name" "$extract_id" "$config" "$SOURCE_COMMIT" >> "$SUBMISSION_LOG"
    echo "Submitted smoke heads modality=$modality job_id=$heads_id (afterok:$extract_id)"
done

echo "packed30 smoke plan: train jobs=$TRAIN_JOBS chains=3 run_id=$RUN_ID source_commit=$SOURCE_COMMIT"
if [ "$DRY_RUN" = "1" ]; then
    if [ "$TRAIN_JOBS" -ne 3 ]; then
        echo "DRY RUN expected exactly 3 smoke trainings; found $TRAIN_JOBS." >&2
        exit 1
    fi
    echo "DRY RUN complete: no jobs submitted."
else
    echo "Submitted smoke jobs. Registry: $SUBMISSION_LOG"
    echo "Smoke scores have no scientific pass threshold and must not change the locked protocol."
fi
