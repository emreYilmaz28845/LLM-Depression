#!/usr/bin/env bash
set -euo pipefail

# Submits the English-translated training matrix (fold CV) through the
# canonical submit_train_and_eval.sh wrapper. English manifests are already
# built, so manifest rebuilds are skipped (SKIP_MANIFEST_BUILD=1).
#
# Usage: FOLDS="0" bash scripts/submit_en_translation_matrix.sh   # smoke
#        FOLDS="0 1 2 3 4" bash scripts/submit_en_translation_matrix.sh
# Env: PROJECT_ROOT, ENV_ACTIVATE, SBATCH_JOB_NAME_PREFIX

PROJECT_ROOT="${PROJECT_ROOT:-/gpfs/projects/etur92/ozu647717/AudioLLM/LLM-Depression}"
ENV_ACTIVATE="${ENV_ACTIVATE:-/gpfs/projects/etur92/ozu647717/venvs/qwen_mn5_rebuilt/bin/activate}"
FOLDS="${FOLDS:-0 1 2 3 4}"
JOB_PREFIX="${JOB_PREFIX:-en-}"
SKIP_MANIFEST_BUILD="${SKIP_MANIFEST_BUILD:-1}"

# D3TEC / ANDROIDS dataset roots must be visible to the compute jobs.
export D3TEC_DATASET_ROOT="${D3TEC_DATASET_ROOT:-/gpfs/projects/etur92/ozu647717/AudioLLM/Datasets/D3TEC DATASET/D3TEC DATASET}"
export D3TEC_FULL_TRANSCRIPTS="$D3TEC_DATASET_ROOT/transcripts_qwen3_asr_spanish.jsonl"
export D3TEC_SEGMENT_TRANSCRIPTS="$D3TEC_DATASET_ROOT/transcripts_qwen3_asr_spanish_segments.jsonl"
export ANDROIDS_DATASET_ROOT="${ANDROIDS_DATASET_ROOT:-/gpfs/projects/etur92/ozu647717/AudioLLM/Datasets/Androids-Corpus/Androids-Corpus}"
export ANDROIDS_INTERVIEW_FULL_TRANSCRIPTS="$ANDROIDS_DATASET_ROOT/interview_transcripts_qwen3_asr_italian.jsonl"
export ANDROIDS_INTERVIEW_SEGMENT_TRANSCRIPTS="$ANDROIDS_DATASET_ROOT/interview_transcripts_qwen3_asr_italian_segments.jsonl"

if [ -f "$ENV_ACTIVATE" ]; then
    # shellcheck disable=SC1090
    source "$ENV_ACTIVATE"
fi

EN_DIR="$PROJECT_ROOT/configs/experiments/translation_en"
MATRIX=(
    "$EN_DIR/cmdc_en_audio_only_selposf1_tf.yaml|en_cmdc_audio_only_v1"
    "$EN_DIR/cmdc_en_audio_text_selposf1_tf.yaml|en_cmdc_audio_text_v1"
    "$EN_DIR/cmdc_en_text_only_selposf1_tf.yaml|en_cmdc_text_only_v1"
    "$EN_DIR/turkish_t17_en_audio_only_selposf1_tf_qwen3asr.yaml|en_turkish_audio_only_v1"
    "$EN_DIR/turkish_t17_en_audio_text_selposf1_tf_qwen3asr.yaml|en_turkish_audio_text_v1"
    "$EN_DIR/turkish_t17_en_text_only_selposf1_tf_qwen3asr.yaml|en_turkish_text_only_v1"
    "$EN_DIR/d3tec_en_audio_only_rotary.yaml|en_d3tec_audio_only_v1"
    "$EN_DIR/d3tec_en_audio_text_rotary.yaml|en_d3tec_audio_text_v1"
    "$EN_DIR/d3tec_en_text_only.yaml|en_d3tec_text_only_v1"
    "$EN_DIR/androids_interview_en_audio_only.yaml|en_androids_audio_only_v1"
    "$EN_DIR/androids_interview_en_audio_text_segment_aligned.yaml|en_androids_audio_text_v1"
    "$EN_DIR/androids_interview_en_text_only.yaml|en_androids_text_only_v1"
)

for entry in "${MATRIX[@]}"; do
    CONFIG="${entry%%|*}"
    RUN_NAME="${entry##*|}"
    for FOLD in $FOLDS; do
        echo "Submitting $RUN_NAME fold=$FOLD (config $CONFIG)"
        CONFIG="$CONFIG" RUN_NAME="$RUN_NAME" FOLD="$FOLD" SKIP_MANIFEST_BUILD="$SKIP_MANIFEST_BUILD" \
            SBATCH_JOB_NAME="${JOB_PREFIX}${RUN_NAME}-f${FOLD}" \
            bash "$PROJECT_ROOT/scripts/submit_train_and_eval.sh"
    done
done
echo "All submissions issued."
