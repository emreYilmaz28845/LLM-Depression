#!/bin/bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/gpfs/projects/etur92/ozu647717/AudioLLM/LLM-Depression}"
ENV_ACTIVATE="${ENV_ACTIVATE:-/gpfs/projects/etur92/ozu647717/venvs/qwen_mn5_rebuilt/bin/activate}"
MATRIX_CONFIG="${MATRIX_CONFIG:-$PROJECT_ROOT/configs/features/androids_hidden_matrix.yaml}"
DRY_RUN="${DRY_RUN:-1}"
RUN_ID="${RUN_ID:-androids_hidden_$(date -u +%Y%m%dT%H%M%SZ)}"
SOURCE_COMMIT="${SOURCE_COMMIT:?Pass the tested local source commit explicitly.}"
SOURCE_RUN_ID="${SOURCE_RUN_ID:-androids_interview_prod_20260730T145948Z}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-$PROJECT_ROOT/output_model/experiments/androids_interview}"
MANIFEST_PATH="${MANIFEST_PATH:-$PROJECT_ROOT/outputs/manifests_androids_interview/androids_interview_manifest.jsonl}"
SPLIT_PATH="${SPLIT_PATH:-$PROJECT_ROOT/outputs/splits_androids_interview/androids_interview_folds.json}"
MANIFEST_HASH="${MANIFEST_HASH:-01a351f7277e4763a8bb9e4983bba190b265becafafca6d7ee04bdcfc948cbed}"
SPLIT_HASH="${SPLIT_HASH:-f75dd2ba7bb324af26de8c5ae3497d2108e6b50815c0ef6cbcade7de70992518}"
DATASET_BASE_ROOT="${DATASET_BASE_ROOT:-/gpfs/projects/etur92/ozu647717/AudioLLM/Datasets}"
REGISTRY="${REGISTRY:-$PROJECT_ROOT/outputs/androids_hidden_jobs/$RUN_ID.tsv}"
SMOKE_AUDIT_PATH="${SMOKE_AUDIT_PATH:-}"

EXTRACT_WORKER="$PROJECT_ROOT/scripts/run_androids_hidden_extract_slurm.sh"
FIXED_WORKER="$PROJECT_ROOT/scripts/run_androids_hidden_fixed_slurm.sh"
OPTUNA_WORKER="$PROJECT_ROOT/scripts/run_androids_hidden_optuna_slurm.sh"
AUDIT_WORKER="$PROJECT_ROOT/scripts/run_androids_hidden_audit_slurm.sh"

if [[ "$DRY_RUN" != 0 && "$DRY_RUN" != 1 ]]; then
    echo "DRY_RUN must be 0 or 1." >&2
    exit 1
fi
if [ ! -s "$MATRIX_CONFIG" ]; then
    echo "Missing Androids hidden matrix config: $MATRIX_CONFIG" >&2
    exit 1
fi
for path in "$EXTRACT_WORKER" "$FIXED_WORKER" "$OPTUNA_WORKER" "$AUDIT_WORKER"; do
    [ -s "$path" ] || { echo "Missing worker: $path" >&2; exit 1; }
done
if [ "$DRY_RUN" = 0 ]; then
    [ -s "$SMOKE_AUDIT_PATH" ] || { echo "Production requires a passed smoke audit." >&2; exit 1; }
    python - "$SMOKE_AUDIT_PATH" <<'PY'
import json, sys
payload = json.load(open(sys.argv[1], encoding="utf-8"))
if payload.get("status") != "passed" or payload.get("mode") != "smoke":
    raise SystemExit("Smoke acceptance is not passed.")
PY
    if [ -e "$REGISTRY" ]; then
        echo "Refusing colliding Androids hidden job registry: $REGISTRY" >&2
        exit 1
    fi
    mkdir -p "$(dirname "$REGISTRY")"
    printf 'timestamp_utc\tjob_type\tjob_id\tdependency\tmodality\tfold\toutput_path\tsource_commit\tchecksum_manifest\n' > "$REGISTRY"
fi

if [ "$DRY_RUN" = 0 ] || [ "${RUN_PREFLIGHT:-1}" = 1 ]; then
    # shellcheck disable=SC1090
    source "$ENV_ACTIVATE"
    cd "$PROJECT_ROOT"
    export PYTHONPATH="$PROJECT_ROOT/.deps/qwen_hidden:$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
    python - "$MATRIX_CONFIG" "$MANIFEST_PATH" "$SPLIT_PATH" "$MANIFEST_HASH" "$SPLIT_HASH" <<'PY'
import hashlib, json, sys
from pathlib import Path
import yaml
from src.data.runtime import load_manifest_rows
from src.utils import sha256_jsonl_rows

matrix = yaml.safe_load(Path(sys.argv[1]).read_text(encoding="utf-8"))
if matrix.get("dataset") != "androids_interview" or matrix.get("fixed_heads") != ["logreg_raw", "xgb_raw"]:
    raise SystemExit("Androids hidden matrix config is not the frozen protocol.")
if matrix.get("optuna", {}).get("target_trials") != 150 or matrix.get("optuna", {}).get("inner_folds") != 3:
    raise SystemExit("Androids hidden Optuna matrix config has the wrong trial protocol.")
manifest_rows = load_manifest_rows(Path(sys.argv[2]))
if sha256_jsonl_rows(manifest_rows) != sys.argv[4]:
    raise SystemExit("Androids manifest canonical hash mismatch.")
digest = hashlib.sha256(Path(sys.argv[3]).read_bytes()).hexdigest()
if digest != sys.argv[5]:
    raise SystemExit("Androids official split hash mismatch.")
PY
    df -h "$PROJECT_ROOT"
python - <<'PY'
import platform
import optuna
import sklearn
import xgboost

versions = {
    "python": platform.python_version(),
    "optuna": optuna.__version__,
    "xgboost": xgboost.__version__,
    "sklearn": sklearn.__version__,
}
expected = {"python": "3.10.14", "optuna": "4.4.0", "xgboost": "2.1.4", "sklearn": "1.7.0"}
print(versions)
if versions != expected:
    raise SystemExit(f"Androids hidden environment mismatch: expected={expected} observed={versions}")
PY
fi

CACHE_ROOT="$PROJECT_ROOT/outputs/hidden_features/androids_interview/$RUN_ID"
CLASSIFIER_ROOT="$PROJECT_ROOT/outputs/hidden_classifiers/androids_interview/$RUN_ID"
AUDIT_OUT="$PROJECT_ROOT/outputs/androids_hidden_audits/$RUN_ID/acceptance.json"
mkdir -p "$CACHE_ROOT" "$CLASSIFIER_ROOT"

if [ "$DRY_RUN" = 0 ]; then
    SOURCE_COMMIT="$SOURCE_COMMIT"
fi

declare -a ALL_CLASSIFIER_IDS=()
EXTRACTION_COUNT=0
FIXED_COUNT=0
OPTUNA_COUNT=0

record_job() {
    local job_type="$1" job_id="$2" dependency="$3" modality="$4" fold="$5" output="$6" checksum="$7"
    if [ "$DRY_RUN" = 0 ]; then
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
            "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$job_type" "$job_id" "$dependency" "$modality" "$fold" "$output" "$SOURCE_COMMIT" "$checksum" >> "$REGISTRY"
    fi
}

submit_job() {
    local name="$1" dependency="$2" export_spec="$3" worker="$4"
    if [ "$DRY_RUN" = 1 ]; then
        echo "DRY RUN sbatch --job-name=$name --dependency=${dependency:-none} --export=$export_spec $worker"
        return 0
    fi
    local dependency_args=()
    if [ -n "$dependency" ]; then dependency_args=(--dependency="afterok:$dependency"); fi
    local raw
    raw="$(sbatch --parsable --job-name="$name" "${dependency_args[@]}" --export="ALL,$export_spec" "$worker")"
    printf '%s' "${raw%%;*}"
}

for modality in audio_only audio_text text_only; do
    case "$modality" in
        audio_only) source_subdir=audio_only; source_condition=audio_only ;;
        audio_text) source_subdir=audio_text_segment_aligned; source_condition=audio_text_segment_aligned ;;
        text_only) source_subdir=text_only; source_condition=text_only ;;
    esac
    source_run_name="${SOURCE_RUN_ID}_androids_interview_${source_subdir}"
    previous_extract=""
    for fold in 0 1 2 3 4; do
        checkpoint="$CHECKPOINT_ROOT/$source_subdir/$source_run_name/fold_$fold/best_model"
        cache_dir="$CACHE_ROOT/$modality/fold_$fold"
        fixed_root="$CLASSIFIER_ROOT/$modality/fold_$fold"
        optuna_dir="$fixed_root/xgb_optuna_150t_d6"
        extract_id=""
        if [ "$DRY_RUN" = 0 ]; then
            [ -s "$checkpoint/adapter_model.safetensors" ] || { echo "Missing checkpoint: $checkpoint" >&2; exit 1; }
            [ -s "$checkpoint/adapter_config.json" ] || { echo "Missing checkpoint config: $checkpoint" >&2; exit 1; }
            [ ! -e "$cache_dir" ] || { echo "Refusing cache collision: $cache_dir" >&2; exit 1; }
            [ ! -e "$fixed_root" ] || { echo "Refusing classifier collision: $fixed_root" >&2; exit 1; }
        fi
        extract_export="PROJECT_ROOT=$PROJECT_ROOT,ENV_ACTIVATE=$ENV_ACTIVATE,RUN_ID=$RUN_ID,MODALITY=$modality,FOLD=$fold,CHECKPOINT_DIR=$checkpoint,OUTPUT_DIR=$cache_dir,MANIFEST_PATH=$MANIFEST_PATH,SOURCE_COMMIT=$SOURCE_COMMIT,SOURCE_RUN_ID=$SOURCE_RUN_ID"
        if [ "$DRY_RUN" = 1 ]; then
            extract_id="DRY_EXT_${modality}_${fold}"
            echo "DRY RUN extraction modality=$modality fold=$fold checkpoint=$checkpoint output=$cache_dir"
        else
            extract_id="$(submit_job "and-he-${modality:0:8}-f${fold}" "$previous_extract" "$extract_export" "$EXTRACT_WORKER")"
            echo "Submitted extraction modality=$modality fold=$fold job_id=$extract_id"
        fi
        record_job extract "$extract_id" "${previous_extract:+afterok:$previous_extract}" "$modality" "$fold" "$cache_dir" "$cache_dir/cache_sha256.tsv"
        EXTRACTION_COUNT=$((EXTRACTION_COUNT + 1))
        previous_extract="$extract_id"

        fixed_export="PROJECT_ROOT=$PROJECT_ROOT,ENV_ACTIVATE=$ENV_ACTIVATE,RUN_ID=$RUN_ID,MODALITY=$modality,FOLD=$fold,CACHE_DIR=$cache_dir,OUTPUT_ROOT=$fixed_root,SOURCE_COMMIT=$SOURCE_COMMIT,SEED=1337"
        if [ "$DRY_RUN" = 1 ]; then
            fixed_id="DRY_FIX_${modality}_${fold}"
            echo "DRY RUN fixed modality=$modality fold=$fold dependency=$extract_id output=$fixed_root"
        else
            fixed_id="$(submit_job "and-hf-${modality:0:8}-f${fold}" "$extract_id" "$fixed_export" "$FIXED_WORKER")"
            echo "Submitted fixed modality=$modality fold=$fold job_id=$fixed_id"
        fi
        record_job fixed "$fixed_id" "afterok:$extract_id" "$modality" "$fold" "$fixed_root" "$fixed_root/artifact_sha256.tsv"
        FIXED_COUNT=$((FIXED_COUNT + 1))
        ALL_CLASSIFIER_IDS+=("$fixed_id")

        optuna_export="PROJECT_ROOT=$PROJECT_ROOT,ENV_ACTIVATE=$ENV_ACTIVATE,RUN_ID=$RUN_ID,MODALITY=$modality,FOLD=$fold,CACHE_DIR=$cache_dir,OUTPUT_DIR=$optuna_dir,SOURCE_COMMIT=$SOURCE_COMMIT,TARGET_TRIALS=150,INNER_FOLDS=3,SEED=1337,INNER_SEED=1337,XGB_THREADS=20"
        if [ "$DRY_RUN" = 1 ]; then
            optuna_id="DRY_OPT_${modality}_${fold}"
            echo "DRY RUN optuna modality=$modality fold=$fold dependency=$extract_id output=$optuna_dir"
        else
            optuna_id="$(submit_job "and-ho-${modality:0:8}-f${fold}" "$extract_id" "$optuna_export" "$OPTUNA_WORKER")"
            echo "Submitted Optuna modality=$modality fold=$fold job_id=$optuna_id"
        fi
        record_job optuna "$optuna_id" "afterok:$extract_id" "$modality" "$fold" "$optuna_dir" "$optuna_dir/artifact_sha256.tsv"
        OPTUNA_COUNT=$((OPTUNA_COUNT + 1))
        ALL_CLASSIFIER_IDS+=("$optuna_id")
    done
done

if [ "$DRY_RUN" = 1 ]; then
    dependency="$(IFS=:; printf '%s' "${ALL_CLASSIFIER_IDS[*]}")"
    echo "DRY RUN audit dependency=$dependency output=$AUDIT_OUT"
    echo "ANDROIDS hidden matrix plan: extraction=$EXTRACTION_COUNT fixed=$FIXED_COUNT optuna=$OPTUNA_COUNT audit=1 total=46 run_id=$RUN_ID"
    exit 0
fi

audit_dependency="$(IFS=:; printf '%s' "${ALL_CLASSIFIER_IDS[*]}")"
audit_export="PROJECT_ROOT=$PROJECT_ROOT,ENV_ACTIVATE=$ENV_ACTIVATE,MODE=production,RUN_ID=$RUN_ID,SOURCE_COMMIT=$SOURCE_COMMIT,AUDIT_OUT=$AUDIT_OUT,JOB_REGISTRY=$REGISTRY,CACHE_ROOT=$CACHE_ROOT,CLASSIFIER_ROOT=$CLASSIFIER_ROOT"
audit_id="$(submit_job "and-hid-audit" "$audit_dependency" "$audit_export" "$AUDIT_WORKER")"
record_job audit "$audit_id" "afterok:$audit_dependency" all - "$AUDIT_OUT" "$AUDIT_OUT"
echo "Submitted Androids hidden final audit job_id=$audit_id registry=$REGISTRY"
echo "ANDROIDS hidden matrix submitted: extraction=$EXTRACTION_COUNT fixed=$FIXED_COUNT optuna=$OPTUNA_COUNT audit=1 total=46 run_id=$RUN_ID"
