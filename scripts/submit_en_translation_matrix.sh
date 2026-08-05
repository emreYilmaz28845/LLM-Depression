#!/usr/bin/env bash
set -euo pipefail

# Launches one sequential CV chain per dataset for the English-translation
# training matrix. Each chain trains one fold at a time (4 GPUs per chain),
# chaining train -> best-checkpoint eval -> next fold via job dependencies,
# and writes a per-config CV summary when a config's folds complete.
#
# Usage: bash scripts/submit_en_translation_matrix.sh
# Env: PROJECT_ROOT, ENV_ACTIVATE, FOLDS, JOB_PREFIX, SKIP_MANIFEST_BUILD

PROJECT_ROOT="${PROJECT_ROOT:-/gpfs/projects/etur92/ozu647717/AudioLLM/LLM-Depression}"
ENV_ACTIVATE="${ENV_ACTIVATE:-/gpfs/projects/etur92/ozu647717/venvs/qwen_mn5_rebuilt/bin/activate}"
FOLDS="${FOLDS:-0 1 2 3 4}"
JOB_PREFIX="${JOB_PREFIX:-en-seq-}"
SKIP_MANIFEST_BUILD="${SKIP_MANIFEST_BUILD:-1}"
CHAIN_SCRIPT="${CHAIN_SCRIPT:-$PROJECT_ROOT/scripts/run_en_translation_chain.sh}"

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

declare -A DATASET_CONFIGS=(
    [cmdc]="$EN_DIR/cmdc_en_audio_only_selposf1_tf.yaml|$EN_DIR/cmdc_en_audio_text_selposf1_tf.yaml|$EN_DIR/cmdc_en_text_only_selposf1_tf.yaml"
    [turkish]="$EN_DIR/turkish_t17_en_audio_only_selposf1_tf_qwen3asr.yaml|$EN_DIR/turkish_t17_en_audio_text_selposf1_tf_qwen3asr.yaml|$EN_DIR/turkish_t17_en_text_only_selposf1_tf_qwen3asr.yaml"
    [d3tec]="$EN_DIR/d3tec_en_audio_only_rotary.yaml|$EN_DIR/d3tec_en_audio_text_rotary.yaml|$EN_DIR/d3tec_en_text_only.yaml"
    [androids]="$EN_DIR/androids_interview_en_audio_only.yaml|$EN_DIR/androids_interview_en_audio_text_segment_aligned.yaml|$EN_DIR/androids_interview_en_text_only.yaml"
)
declare -A DATASET_RUNS=(
    [cmdc]="en_seq_cmdc_audio_only_v1|en_seq_cmdc_audio_text_v1|en_seq_cmdc_text_only_v1"
    [turkish]="en_seq_turkish_audio_only_v1|en_seq_turkish_audio_text_v1|en_seq_turkish_text_only_v1"
    [d3tec]="en_seq_d3tec_audio_only_v1|en_seq_d3tec_audio_text_v1|en_seq_d3tec_text_only_v1"
    [androids]="en_seq_androids_audio_only_v1|en_seq_androids_audio_text_v1|en_seq_androids_text_only_v1"
)

for DATASET in cmdc turkish d3tec androids; do
    echo "Launching sequential chain for $DATASET"
    sbatch --job-name="${JOB_PREFIX}chain-$DATASET" --export=ALL \
        --export="ALL,PROJECT_ROOT=$PROJECT_ROOT,CONFIG_LIST=${DATASET_CONFIGS[$DATASET]},RUN_NAMES=${DATASET_RUNS[$DATASET]},FOLDS=$FOLDS,INDEX=0,SKIP_MANIFEST_BUILD=$SKIP_MANIFEST_BUILD,JOB_PREFIX=$JOB_PREFIX,CHAIN_SCRIPT=$CHAIN_SCRIPT" \
        "$CHAIN_SCRIPT"
done
echo "4 sequential chains launched (one fold at a time, 4 GPUs per chain)."
