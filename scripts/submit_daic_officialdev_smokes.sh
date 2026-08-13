#!/usr/bin/env bash
# Submit all DAIC official-development smokes.
#
# Smoke scope (isolated roots; never reportable, never in the workbook or
# W&B): six synthetic forward contracts (one per backbone/modality), two
# one-epoch training smokes (Qwen and Gemma audio+text, official-train
# subjects only, inner validation, final dev eval disabled), and six
# extraction/head smoke chains (fit+eval subjects drawn entirely from the
# official training partition, both classes, selection file hashed into the
# cache identity). Production commands never carry the smoke selection.
# Dry-run by default; zero mutation in dry-run mode.
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/gpfs/projects/etur92/ozu647717/AudioLLM/LLM-Depression}"
SMOKE_ID="${SMOKE_ID:?Set a unique SMOKE_ID}"
DRY_RUN="${DRY_RUN:-1}"
SMOKE_ROOT="${SMOKE_ROOT:-$PROJECT_ROOT/outputs/daic_officialdev_smokes}"
CONTRACT_WORKER="${CONTRACT_WORKER:-$PROJECT_ROOT/scripts/run_daic_officialdev_contract_slurm.sh}"
TRAIN_WORKER="${TRAIN_WORKER:-$PROJECT_ROOT/scripts/run_daic_officialdev_smoke_train_slurm.sh}"
EXTRACT_WORKER="${EXTRACT_WORKER:-$PROJECT_ROOT/scripts/run_daic_officialdev_smoke_extract_slurm.sh}"
HEADS_WORKER="${HEADS_WORKER:-$PROJECT_ROOT/scripts/run_daic_officialdev_smoke_heads_slurm.sh}"
QWEN_ENV_ACTIVATE="${QWEN_ENV_ACTIVATE:-/gpfs/projects/etur92/ozu647717/venvs/qwen_mn5_rebuilt/bin/activate}"
GEMMA_ENV_ACTIVATE="${GEMMA_ENV_ACTIVATE:-/gpfs/projects/etur92/ozu647717/venvs/gemma4_12b_tf5_14_1/bin/activate}"
MODEL_PATH_QWEN="${MODEL_PATH_QWEN:-/gpfs/projects/etur92/ozu647717/models/Qwen2-Audio-7B-Instruct}"
MODEL_PATH_QWEN_TEXT="${MODEL_PATH_QWEN_TEXT:-/gpfs/projects/etur92/ozu647717/models/Qwen2-7B-Instruct}"
MODEL_PATH_GEMMA4="${MODEL_PATH_GEMMA4:-/gpfs/projects/etur92/ozu647717/models/gemma-4-12B-it/707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7}"
DAIC_UNPROCESSED_ROOT="${DAIC_UNPROCESSED_ROOT:-/gpfs/projects/etur92/ozu647717/AudioLLM/Datasets/DAIC-WOZ/unprocessed}"
DAIC_LABEL_ROOT="${DAIC_LABEL_ROOT:-/gpfs/projects/etur92/ozu647717/AudioLLM/Datasets/DAIC-WOZ/minimal_zips}"
SMOKE_SUBJECT_LIMIT="${SMOKE_SUBJECT_LIMIT:-6}"

case "$DRY_RUN" in 0|1) ;; *) echo "DRY_RUN must be 0 or 1" >&2; exit 2;; esac
for path in "$CONTRACT_WORKER" "$TRAIN_WORKER" "$EXTRACT_WORKER" "$HEADS_WORKER"; do
    [ -f "$path" ] || { echo "Missing required file: $path" >&2; exit 3; }
done

cd "$PROJECT_ROOT"
SMOKE_RUN_ROOT="$SMOKE_ROOT/$SMOKE_ID"

# Qwen contract parents: completed harmonized DAIC official-test checkpoints.
QWEN_PARENT_AUDIO="$PROJECT_ROOT/output_model/harmonized_v1/audio_only/daic/harmonized_v1_harmonized_v1_prod_20260809T171705Z_d1e8130b_daic_audio_only_r1/fold_0"
QWEN_PARENT_AUDIO_TEXT="$PROJECT_ROOT/output_model/harmonized_v1/audio_text/daic/harmonized_v1_harmonized_v1_prod_20260809T171705Z_d1e8130b_daic_audio_text_r1/fold_0"
QWEN_PARENT_TEXT="$PROJECT_ROOT/output_model/harmonized_v1/text_only/daic/harmonized_v1_harmonized_v1_prod_20260809T171705Z_d1e8130b_daic_text_only_r1/fold_0"
GEMMA_PARENT_AUDIO="$PROJECT_ROOT/output_model/harmonized_v1_gemma4/audio_only/daic/gemma4_harmonized_v1_gemma4_v1_prod_20260812T020449Z_cca3f4ae_daic_audio_only/fold_0"
GEMMA_PARENT_AUDIO_TEXT="$PROJECT_ROOT/output_model/harmonized_v1_gemma4/audio_text/daic/gemma4_harmonized_v1_gemma4_v1_prod_20260812T020449Z_cca3f4ae_daic_audio_text_r2/fold_0"
GEMMA_PARENT_TEXT="$PROJECT_ROOT/output_model/harmonized_v1_gemma4/text_only/daic/gemma4_harmonized_v1_gemma4_v1_prod_20260812T020449Z_cca3f4ae_daic_text_only/fold_0"

submit() {
    if [ "$DRY_RUN" = 1 ]; then
        printf 'DRY_RUN ' >&2; printf '%q ' "$@" >&2; printf '\n' >&2
        printf 'dry_%s\n' "$(printf '%s\0' "$@" | sha256sum | cut -c1-12)"
    else
        "$@"
    fi
}
job_id() { printf '%s' "${1%%;*}"; }

if [ "$DRY_RUN" = 0 ]; then
    mkdir -p "$SMOKE_RUN_ROOT"
    if [ -e "$SMOKE_RUN_ROOT/jobs.tsv" ]; then
        # Resume mode: jobs already recorded for a (kind, backbone, modality)
        # are skipped; Tier C reads the train job ids from the registry.
        echo "Resuming smoke registry: $SMOKE_RUN_ROOT/jobs.tsv"
    else
        printf 'kind\tbackbone\tmodality\tjob_id\tdependency\n' > "$SMOKE_RUN_ROOT/jobs.tsv"
    fi
fi
record_job() {
    [ "$DRY_RUN" = 1 ] || printf '%s\t%s\t%s\t%s\t%s\n' "$1" "$2" "$3" "$4" "$5" >> "$SMOKE_RUN_ROOT/jobs.tsv"
}
already_submitted() {
    [ "$DRY_RUN" = 1 ] && return 1
    awk -F'\t' -v k="$1" -v b="$2" -v m="$3" '$1==k && $2==b && $3==m {found=1} END {exit !found}' "$SMOKE_RUN_ROOT/jobs.tsv"
}

# ---- Tier A: six forward contracts -----------------------------------------
contract_jobs=()
for backbone in qwen gemma4; do
    for modality in audio_only audio_text text_only; do
        if already_submitted contract "$backbone" "$modality"; then
            echo "skip contract $backbone/$modality (already recorded)"
            continue
        fi
        case "$backbone:$modality" in
            qwen:audio_only) MODEL="$MODEL_PATH_QWEN"; ADAPTER="$QWEN_PARENT_AUDIO/best_model";;
            qwen:audio_text) MODEL="$MODEL_PATH_QWEN"; ADAPTER="$QWEN_PARENT_AUDIO_TEXT/best_model";;
            qwen:text_only) MODEL="$MODEL_PATH_QWEN_TEXT"; ADAPTER="$QWEN_PARENT_TEXT/best_model";;
            gemma4:audio_only) MODEL="$MODEL_PATH_GEMMA4"; ADAPTER="$GEMMA_PARENT_AUDIO/best_model";;
            gemma4:audio_text) MODEL="$MODEL_PATH_GEMMA4"; ADAPTER="$GEMMA_PARENT_AUDIO_TEXT/best_model";;
            gemma4:text_only) MODEL="$MODEL_PATH_GEMMA4"; ADAPTER="$GEMMA_PARENT_TEXT/best_model";;
        esac
        if [ "$backbone" = "gemma4" ]; then ENV="$GEMMA_ENV_ACTIVATE"; else ENV="$QWEN_ENV_ACTIVATE"; fi
        [ -d "$ADAPTER" ] || { echo "Missing contract adapter: $ADAPTER" >&2; exit 3; }
        OUTPUT="$SMOKE_RUN_ROOT/contracts/$backbone/$modality"
        RAW="$(submit sbatch --parsable --job-name="od-smk-ct-${backbone:0:3}-${modality:0:2}" \
            --export="ALL,PROJECT_ROOT=$PROJECT_ROOT,ENV_ACTIVATE=$ENV,BACKBONE=$backbone,MODALITY=$modality,MODEL_PATH=$MODEL,ADAPTER_PATH=$ADAPTER,OUTPUT=$OUTPUT" \
            "$CONTRACT_WORKER")"
        JOB="$(job_id "$RAW")"
        contract_jobs+=("$JOB")
        record_job contract "$backbone" "$modality" "$JOB" ""
    done
done

# ---- Tier B: two training smokes -------------------------------------------
train_jobs=()
for backbone in qwen gemma4; do
    if already_submitted train "$backbone" "audio_text"; then
        echo "skip train $backbone (already recorded)"
        continue
    fi
    case "$backbone" in
        qwen)
            CONFIG="$PROJECT_ROOT/configs/main/daic_audio_text_harmonized_selmacrof1_tf_officialdev.yaml"
            ENV="$QWEN_ENV_ACTIVATE"; MODEL="$MODEL_PATH_QWEN"
            ;;
        gemma4)
            CONFIG="$PROJECT_ROOT/configs/main/daic_audio_text_harmonized_selmacrof1_tf_gemma4_12b_officialdev.yaml"
            ENV="$GEMMA_ENV_ACTIVATE"; MODEL="$MODEL_PATH_GEMMA4"
            ;;
    esac
    RUN_NAME="daic_officialdev_smoke_train_${backbone}_audio_text_${SMOKE_ID}"
    TRAIN_ROOT="$SMOKE_RUN_ROOT/train/$backbone"
    RAW="$(submit sbatch --parsable --job-name="od-smk-tr-${backbone:0:3}" \
        --export="ALL,PROJECT_ROOT=$PROJECT_ROOT,ENV_ACTIVATE=$ENV,CONFIG=$CONFIG,RUN_NAME=$RUN_NAME,SMOKE_RUN_ROOT=$TRAIN_ROOT,SMOKE_SUBJECT_LIMIT=$SMOKE_SUBJECT_LIMIT,MODEL_PATH=$MODEL,TEXT_MODEL_PATH=$MODEL_PATH_QWEN_TEXT,GEMMA4_MODEL_PATH=$MODEL_PATH_GEMMA4,DAIC_UNPROCESSED_ROOT=$DAIC_UNPROCESSED_ROOT,DAIC_LABEL_ROOT=$DAIC_LABEL_ROOT" \
        "$TRAIN_WORKER")"
    JOB="$(job_id "$RAW")"
    train_jobs+=("$JOB")
    record_job train "$backbone" "audio_text" "$JOB" ""
done

# ---- Tier C: six extraction/head smoke chains ------------------------------
for backbone in qwen gemma4; do
    for modality in audio_only audio_text text_only; do
        if already_submitted extract "$backbone" "$modality"; then
            echo "skip extract/heads $backbone/$modality (already recorded)"
            continue
        fi
        case "$backbone:$modality" in
            qwen:audio_only) MODEL="$MODEL_PATH_QWEN";;
            qwen:audio_text) MODEL="$MODEL_PATH_QWEN";;
            qwen:text_only) MODEL="$MODEL_PATH_QWEN_TEXT";;
            gemma4:*) MODEL="$MODEL_PATH_GEMMA4";;
        esac
        if [ "$backbone" = "gemma4" ]; then ENV="$GEMMA_ENV_ACTIVATE"; else ENV="$QWEN_ENV_ACTIVATE"; fi
        # The smoke parent is the smoke training run of this backbone. Its
        # saved split carries limited official-train subjects with both
        # classes; the selection file is built by the extract worker at job
        # start (the extract job depends afterok on the train job).
        TRAIN_ROOT="$SMOKE_RUN_ROOT/train/$backbone"
        PARENT_FOLD_DIR="$TRAIN_ROOT/daic_officialdev_smoke_train_${backbone}_audio_text_${SMOKE_ID}/fold_0"
        SELECTION="$SMOKE_RUN_ROOT/selections/${backbone}_${modality}.json"
        ATTEMPT_DIR="$SMOKE_RUN_ROOT/attempts/${backbone}_${modality}"
        SMOKE_NAME="smoke_${backbone}_${modality}_${SMOKE_ID}"
        TRAIN_JOB=""
        if [ -f "$SMOKE_RUN_ROOT/jobs.tsv" ]; then
            TRAIN_JOB="$(awk -F'\t' -v b="$backbone" '$1=="train" && $2==b {print $4; exit}' "$SMOKE_RUN_ROOT/jobs.tsv")"
        fi
        if [ "$DRY_RUN" = 1 ]; then
            TRAIN_JOB="dry_train_${backbone}"
        else
            [ -n "$TRAIN_JOB" ] || { echo "Missing smoke train job for $backbone" >&2; exit 3; }
        fi
        EXTRACT_RAW="$(submit sbatch --parsable --job-name="od-smk-ex-${backbone:0:3}-${modality:0:2}" \
            --dependency="afterok:$TRAIN_JOB" \
            --export="ALL,PROJECT_ROOT=$PROJECT_ROOT,ENV_ACTIVATE=$ENV,ATTEMPT_DIR=$ATTEMPT_DIR,PARENT_FOLD_DIR=$PARENT_FOLD_DIR,MODEL_PATH=$MODEL,CONDITION=daic_officialdev_smoke,SELECTION=$SELECTION" \
            "$EXTRACT_WORKER")"
        EXTRACT_JOB="$(job_id "$EXTRACT_RAW")"
        record_job extract "$backbone" "$modality" "$EXTRACT_JOB" ""
        HEADS_RAW="$(submit sbatch --parsable --job-name="od-smk-hd-${backbone:0:3}-${modality:0:2}" \
            --dependency="afterok:$EXTRACT_JOB" \
            --export="ALL,PROJECT_ROOT=$PROJECT_ROOT,ATTEMPT_DIR=$ATTEMPT_DIR,PARENT_FOLD_DIR=$PARENT_FOLD_DIR,BACKBONE=$backbone,MODALITY=$modality,SMOKE_NAME=$SMOKE_NAME" \
            "$HEADS_WORKER")"
        HEADS_JOB="$(job_id "$HEADS_RAW")"
        record_job heads "$backbone" "$modality" "$HEADS_JOB" "$EXTRACT_JOB"
    done
done

echo "Officialdev smoke plan: id=$SMOKE_ID contracts=6 trains=2 extract=6 heads=6 dry_run=$DRY_RUN"
[ "$DRY_RUN" = 1 ] || echo "Smoke registry: $SMOKE_RUN_ROOT/jobs.tsv"
