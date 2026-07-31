#!/bin/bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/gpfs/projects/etur92/ozu647717/AudioLLM/LLM-Depression}"
ENV_ACTIVATE="${ENV_ACTIVATE:-/gpfs/projects/etur92/ozu647717/venvs/qwen_mn5_rebuilt/bin/activate}"
RUN_ID="${RUN_ID:-androids_hidden_smoke_$(date -u +%Y%m%dT%H%M%SZ)}"
SOURCE_COMMIT="${SOURCE_COMMIT:?Pass the tested local source commit explicitly.}"
SOURCE_RUN_ID="${SOURCE_RUN_ID:-androids_interview_prod_20260730T145948Z}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-$PROJECT_ROOT/output_model/experiments/androids_interview}"
MANIFEST_PATH="${MANIFEST_PATH:-$PROJECT_ROOT/outputs/manifests_androids_interview/androids_interview_manifest.jsonl}"
DRY_RUN="${DRY_RUN:-0}"

if [ "$DRY_RUN" != 0 ]; then
    echo "Smoke submission currently supports only DRY_RUN=0." >&2
    exit 1
fi
# shellcheck disable=SC1090
source "$ENV_ACTIVATE"
cd "$PROJECT_ROOT"
export PROJECT_ROOT
export PYTHONPATH="$PROJECT_ROOT/.deps/qwen_hidden:$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

SMOKE_ROOT="$PROJECT_ROOT/outputs/androids_hidden_smoke/$RUN_ID"
EXTRACTION_DIR="$PROJECT_ROOT/outputs/hidden_features/androids_interview/$RUN_ID/audio_text/fold_0"
SYNTHETIC_DIR="$SMOKE_ROOT/synthetic_cache"
FIXED_ROOT="$SMOKE_ROOT/fixed"
OPTUNA_DIR="$SMOKE_ROOT/optuna/xgb_optuna_150t_d6"
AUDIT_OUT="$SMOKE_ROOT/acceptance.json"
REGISTRY="$PROJECT_ROOT/outputs/androids_hidden_jobs/${RUN_ID}.tsv"
if [ -e "$SMOKE_ROOT" ] || [ -e "$REGISTRY" ]; then
    echo "Refusing smoke collision: $SMOKE_ROOT or $REGISTRY" >&2
    exit 1
fi
mkdir -p "$SMOKE_ROOT" "$(dirname "$REGISTRY")"
python scripts/create_androids_hidden_synthetic_cache.py --output-dir "$SYNTHETIC_DIR" --source-commit "$SOURCE_COMMIT"
printf 'timestamp_utc\tjob_type\tjob_id\tdependency\tmodality\tfold\toutput_path\tsource_commit\tchecksum_manifest\n' > "$REGISTRY"

checkpoint="$CHECKPOINT_ROOT/audio_text_segment_aligned/${SOURCE_RUN_ID}_androids_interview_audio_text_segment_aligned/fold_0/best_model"
extract_export="PROJECT_ROOT=$PROJECT_ROOT,ENV_ACTIVATE=$ENV_ACTIVATE,RUN_ID=$RUN_ID,MODALITY=audio_text,FOLD=0,CHECKPOINT_DIR=$checkpoint,OUTPUT_DIR=$EXTRACTION_DIR,MANIFEST_PATH=$MANIFEST_PATH,SOURCE_COMMIT=$SOURCE_COMMIT,SOURCE_RUN_ID=$SOURCE_RUN_ID,MAX_EXAMPLES=2"
extract_id="$(sbatch --parsable --job-name=and-hid-smoke-ext --export="ALL,$extract_export" scripts/run_androids_hidden_extract_slurm.sh | cut -d';' -f1)"
printf '%s\textract\t%s\t\t%s\t%s\t%s\t%s\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$extract_id" "audio_text" 0 "$EXTRACTION_DIR" "$SOURCE_COMMIT" "$EXTRACTION_DIR/cache_sha256.tsv" >> "$REGISTRY"

fixed_export="PROJECT_ROOT=$PROJECT_ROOT,ENV_ACTIVATE=$ENV_ACTIVATE,RUN_ID=$RUN_ID,MODALITY=audio_only,FOLD=0,CACHE_DIR=$SYNTHETIC_DIR,OUTPUT_ROOT=$FIXED_ROOT,SOURCE_COMMIT=$SOURCE_COMMIT,SEED=1337"
fixed_id="$(sbatch --parsable --job-name=and-hid-smoke-fix --dependency="afterok:$extract_id" --export="ALL,$fixed_export" scripts/run_androids_hidden_fixed_slurm.sh | cut -d';' -f1)"
printf '%s\tfixed\t%s\tafterok:%s\t%s\t%s\t%s\t%s\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$fixed_id" "$extract_id" "audio_only" 0 "$FIXED_ROOT" "$SOURCE_COMMIT" "$FIXED_ROOT/artifact_sha256.tsv" >> "$REGISTRY"

optuna_export="PROJECT_ROOT=$PROJECT_ROOT,ENV_ACTIVATE=$ENV_ACTIVATE,RUN_ID=$RUN_ID,MODALITY=audio_only,FOLD=0,CACHE_DIR=$SYNTHETIC_DIR,OUTPUT_DIR=$OPTUNA_DIR,SOURCE_COMMIT=$SOURCE_COMMIT,TARGET_TRIALS=2,INNER_FOLDS=3,SEED=1337,INNER_SEED=1337,XGB_THREADS=20"
optuna1_id="$(sbatch --parsable --job-name=and-hid-smoke-opt1 --dependency="afterok:$extract_id" --export="ALL,$optuna_export" scripts/run_androids_hidden_optuna_slurm.sh | cut -d';' -f1)"
printf '%s\toptuna\t%s\tafterok:%s\t%s\t%s\t%s\t%s\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$optuna1_id" "$extract_id" "audio_only" 0 "$OPTUNA_DIR" "$SOURCE_COMMIT" "$OPTUNA_DIR/artifact_sha256.tsv" >> "$REGISTRY"
optuna2_id="$(sbatch --parsable --job-name=and-hid-smoke-opt2 --dependency="afterok:$optuna1_id" --export="ALL,$optuna_export" scripts/run_androids_hidden_optuna_slurm.sh | cut -d';' -f1)"
printf '%s\toptuna_resume\t%s\tafterok:%s\t%s\t%s\t%s\t%s\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$optuna2_id" "$optuna1_id" "audio_only" 0 "$OPTUNA_DIR" "$SOURCE_COMMIT" "$OPTUNA_DIR/artifact_sha256.tsv" >> "$REGISTRY"

audit_export="PROJECT_ROOT=$PROJECT_ROOT,ENV_ACTIVATE=$ENV_ACTIVATE,MODE=smoke,RUN_ID=$RUN_ID,SOURCE_COMMIT=$SOURCE_COMMIT,AUDIT_OUT=$AUDIT_OUT,JOB_REGISTRY=$REGISTRY,SMOKE_EXTRACTION_DIR=$EXTRACTION_DIR,SMOKE_FIXED_ROOT=$FIXED_ROOT,SMOKE_OPTUNA_DIR=$OPTUNA_DIR"
audit_id="$(sbatch --parsable --job-name=and-hid-smoke-audit --dependency="afterok:$extract_id:$fixed_id:$optuna2_id" --export="ALL,$audit_export" scripts/run_androids_hidden_audit_slurm.sh | cut -d';' -f1)"
printf '%s\taudit\t%s\tafterok:%s:%s:%s\tall\t-\t%s\t%s\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$audit_id" "$extract_id" "$fixed_id" "$optuna2_id" "$AUDIT_OUT" "$SOURCE_COMMIT" "$AUDIT_OUT" >> "$REGISTRY"
echo "Submitted Androids hidden smoke: extraction=$extract_id fixed=$fixed_id optuna1=$optuna1_id optuna2=$optuna2_id audit=$audit_id run_id=$RUN_ID registry=$REGISTRY"
