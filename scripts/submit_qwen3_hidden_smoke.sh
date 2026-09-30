#!/usr/bin/env bash
# Explicit, evidence-only Qwen3 hidden-extraction smoke submitter.
#
# This is the only route that dispatches Qwen3 hidden-extraction / fixed-head
# jobs. The harmonized production launchers keep `fixed_heads: []` and refuse
# Qwen3 head dispatch, so a production submission can never expand into Qwen3
# head jobs by accident.
#
# Per job in the explicit matrix this script:
#   1. verifies the checkpoint is a fold `best_model` adapter whose saved
#      run_config declares the matrix backend, the recorded GPU shape
#      (`resources.eval_gpus_per_node`), a resolvable base model and an
#      evaluation view;
#   2. refuses existing cache / fit output paths (no overwrite);
#   3. builds (or verifies / previews) the deterministic isolated-smoke subject
#      selection that the extraction hashes into its cache identity;
#   4. in submit mode, submits ONE GPU extraction job (SKIP_CLASSIFIERS=1) and
#      ONE CPU-only classifier job that runs after it (afterok).
#
# Required environment:
#   PROJECT_ROOT         code root (deployment `code` dir or permanent checkout)
#   EVIDENCE_ROOT        writable smoke evidence root (caches, fits, logs)
#   DRY_RUN              1 (default) or 0
#   QWEN3_HEADS_ENABLED  1 required when DRY_RUN=0 (explicit opt-in)
# Optional:
#   MATRIX, QWEN_HIDDEN_DEPS, CLASSIFIER_WORKER, SELECTION_BUILDER
#
# Run it in the project environment (module load + `source <env>/bin/activate`).
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:?Set PROJECT_ROOT to the code root}"
MATRIX="${MATRIX:-$PROJECT_ROOT/configs/features/qwen3_hidden_smoke_matrix.yaml}"
EVIDENCE_ROOT="${EVIDENCE_ROOT:?Set EVIDENCE_ROOT to the writable smoke evidence root}"
DRY_RUN="${DRY_RUN:-1}"
QWEN3_HEADS_ENABLED="${QWEN3_HEADS_ENABLED:-0}"
QWEN_HIDDEN_DEPS="${QWEN_HIDDEN_DEPS:-/gpfs/projects/etur92/ozu647717/AudioLLM/LLM-Depression/.deps/qwen_hidden}"
CLASSIFIER_WORKER="${CLASSIFIER_WORKER:-$PROJECT_ROOT/scripts/run_qwen_hidden_classifier_slurm.sh}"
SELECTION_BUILDER="${SELECTION_BUILDER:-$PROJECT_ROOT/scripts/build_qwen3_smoke_subject_selection.py}"

case "$DRY_RUN" in 0|1) ;; *) echo "DRY_RUN must be 0 or 1" >&2; exit 2;; esac
if [ "$DRY_RUN" = "0" ] && [ "$QWEN3_HEADS_ENABLED" != "1" ]; then
    echo "Refusing: Qwen3 head execution is explicit-only. Set QWEN3_HEADS_ENABLED=1 to submit." >&2
    exit 2
fi
for path in "$MATRIX" "$SELECTION_BUILDER" "$CLASSIFIER_WORKER"; do
    [ -f "$path" ] || { echo "Missing required file: $path" >&2; exit 3; }
done

cd "$PROJECT_ROOT"

SOURCE_COMMIT=""
if [ -f "$PROJECT_ROOT/.provenance/git_commit.txt" ]; then
    SOURCE_COMMIT="$(cat "$PROJECT_ROOT/.provenance/git_commit.txt")"
elif git -C "$PROJECT_ROOT" rev-parse HEAD >/dev/null 2>&1; then
    SOURCE_COMMIT="$(git -C "$PROJECT_ROOT" rev-parse HEAD)"
fi
DEPLOYMENT_ID=""
DEPLOYMENT_JSON="$(dirname "$PROJECT_ROOT")/deployment.json"
if [ -f "$DEPLOYMENT_JSON" ]; then
    DEPLOYMENT_ID="$(python - "$DEPLOYMENT_JSON" <<'PY'
import json, sys
print(json.load(open(sys.argv[1], encoding="utf-8")).get("deployment_id", ""))
PY
)"
fi

TASKS_FILE="$(mktemp)"
if ! python - "$MATRIX" "$PROJECT_ROOT" "$EVIDENCE_ROOT" >"$TASKS_FILE" <<'PY'
import sys
from pathlib import Path

import yaml

root = Path(sys.argv[2])
sys.path.insert(0, str(root))
from src.utils import load_yaml_with_overrides, resolve_model_backend, sha256_file  # noqa: E402

matrix_path = Path(sys.argv[1])
evidence_root = Path(sys.argv[3])
matrix = yaml.safe_load(matrix_path.read_text(encoding="utf-8"))
if matrix.get("schema_version") != "audiollm.qwen3_hidden_smoke.v1":
    raise SystemExit(f"unexpected smoke matrix schema: {matrix.get('schema_version')!r}")
jobs = matrix.get("jobs") or []
if not jobs:
    raise SystemExit("smoke matrix has no jobs")
seen: set[str] = set()
for item in jobs:
    name = str(item["name"])
    if name in seen:
        raise SystemExit(f"duplicate smoke job name: {name}")
    seen.add(name)
    declared_backend = str(item.get("backend") or "")
    if declared_backend not in ("qwen38", "qwen3omni"):
        raise SystemExit(f"{name}: backend must be qwen38 or qwen3omni, got {declared_backend!r}")
    config_path = root / str(item["config"])
    if not config_path.is_file():
        raise SystemExit(f"{name}: missing config: {config_path}")
    config_backend = str(resolve_model_backend(load_yaml_with_overrides(config_path, [])) or "")
    if config_backend != declared_backend:
        raise SystemExit(
            f"{name}: matrix backend {declared_backend!r} != config backend {config_backend!r}"
        )
    checkpoint = Path(str(item["checkpoint_dir"]))
    if checkpoint.name != "best_model":
        raise SystemExit(f"{name}: checkpoint must be a fold best_model directory: {checkpoint}")
    for required in ("adapter_config.json", "adapter_model.safetensors"):
        if not (checkpoint / required).is_file():
            raise SystemExit(f"{name}: checkpoint is missing {required}: {checkpoint}")
    fold_dir = checkpoint.parent
    run_config_path = fold_dir / "run_config.yaml"
    split_path = fold_dir / "logs" / "split_used.json"
    if not run_config_path.is_file():
        raise SystemExit(f"{name}: parent run_config.yaml is missing: {run_config_path}")
    if not split_path.is_file():
        raise SystemExit(f"{name}: saved logs/split_used.json is missing: {split_path}")
    saved = yaml.safe_load(run_config_path.read_text(encoding="utf-8"))
    saved_config = saved.get("config") or {}
    saved_backend = str(saved_config.get("model_backend") or "")
    if saved_backend != declared_backend:
        raise SystemExit(f"{name}: saved run backend {saved_backend!r} != {declared_backend!r}")
    evaluation = saved_config.get("evaluation") or {}
    if not str(evaluation.get("evaluation_view") or "").strip():
        raise SystemExit(f"{name}: saved run has no evaluation.evaluation_view")
    base_model = Path(str(saved.get("resolved_model_name_or_path") or saved_config.get("model_name_or_path") or ""))
    if not (base_model / "config.json").is_file():
        raise SystemExit(f"{name}: resolved base model snapshot is unavailable: {base_model}")
    resources = saved_config.get("resources") or {}
    recorded_gpus = int(resources.get("eval_gpus_per_node", 1) or 1)
    pinned_gpus = int(item.get("gpus", recorded_gpus))
    if pinned_gpus != recorded_gpus:
        raise SystemExit(
            f"{name}: matrix gpus={pinned_gpus} != recorded eval_gpus_per_node={recorded_gpus}"
        )
    cache_dir = evidence_root / name / "cache"
    classifier_dir = evidence_root / name / "classifiers"
    selection_path = evidence_root / name / "subject_selection.json"
    for path in (cache_dir, classifier_dir):
        if path.exists():
            raise SystemExit(f"{name}: refusing existing output path: {path}")
    print("\t".join((
        name,
        declared_backend,
        str(item["condition"]),
        str(config_path),
        str(checkpoint),
        str(run_config_path),
        sha256_file(run_config_path),
        sha256_file(checkpoint / "adapter_model.safetensors"),
        str(cache_dir),
        str(classifier_dir),
        str(selection_path),
        str(pinned_gpus),
        str(int(item["train_per_label"])),
        str(int(item["eval_per_label"])),
    )))
PY
then
    echo "Qwen3 smoke matrix validation failed; refusing to submit: $MATRIX" >&2
    rm -f "$TASKS_FILE"
    exit 5
fi
mapfile -t TASKS < "$TASKS_FILE"
rm -f "$TASKS_FILE"

submit() {
    if [ "$DRY_RUN" = 1 ]; then
        printf 'DRY_RUN ' >&2; printf '%q ' "$@" >&2; printf '\n' >&2
        printf 'dry_%s\n' "$(printf '%s\0' "$@" | sha256sum | cut -c1-12)"
    else
        "$@"
    fi
}
job_id() { printf '%s' "${1%%;*}"; }

registry="$EVIDENCE_ROOT/jobs.tsv"
if [ "$DRY_RUN" = 0 ]; then
    [ ! -e "$registry" ] || { echo "Refusing existing smoke registry: $registry" >&2; exit 4; }
    mkdir -p "$EVIDENCE_ROOT"
    printf 'job\tkind\tslurm_job_id\tdependency\tcheckpoint\tcache_dir\tclassifier_dir\n' > "$registry"
fi

echo "Qwen3 hidden-extraction smoke plan"
echo "  project root: $PROJECT_ROOT"
echo "  source commit: ${SOURCE_COMMIT:-unknown}"
echo "  deployment id: ${DEPLOYMENT_ID:-none}"
echo "  matrix: $MATRIX"
echo "  evidence root: $EVIDENCE_ROOT"
echo "  dry_run: $DRY_RUN (QWEN3_HEADS_ENABLED=$QWEN3_HEADS_ENABLED)"
echo "  jobs: $(( ${#TASKS[@]} * 2 )) ($(( ${#TASKS[@]} )) extraction + $(( ${#TASKS[@]} )) classifier)"

for task in "${TASKS[@]}"; do
    IFS=$'\t' read -r name backend condition config_path checkpoint run_config_path run_config_sha adapter_sha cache_dir classifier_dir selection_path gpus train_per_label eval_per_label <<< "$task"
    backend_vars="$(bash "$PROJECT_ROOT/scripts/harmonized_backend_env.sh" "$config_path" "$PROJECT_ROOT")"
    eval "$backend_vars"
    if [ "$MODEL_BACKEND" != "$backend" ]; then
        echo "$name: backend env resolution '$MODEL_BACKEND' != '$backend'" >&2
        exit 6
    fi
    log_root="$EVIDENCE_ROOT/logs/$name"
    echo "--- job $name ---"
    echo "  backend: $backend"
    echo "  condition: $condition"
    echo "  checkpoint: $checkpoint"
    echo "  run_config: $run_config_path (sha256 $run_config_sha)"
    echo "  adapter_model.safetensors sha256: $adapter_sha"
    echo "  gpus: $gpus (recorded eval_gpus_per_node)"
    echo "  worker env: $ENV_ACTIVATE"
    echo "  cache: $cache_dir"
    echo "  fit output: $classifier_dir"
    echo "  subject selection: $selection_path (train_per_label=$train_per_label eval_per_label=$eval_per_label)"
    echo "  logs: $log_root"
    echo "  classifier variants: $CLASSIFIER_VARIANTS"

    selection_flags=(--checkpoint-dir "$checkpoint" --train-per-label "$train_per_label" --eval-per-label "$eval_per_label")
    if [ "$DRY_RUN" = 1 ]; then
        python "$SELECTION_BUILDER" "${selection_flags[@]}" --preview
    elif [ -f "$selection_path" ]; then
        python "$SELECTION_BUILDER" "${selection_flags[@]}" --output "$selection_path" --verify
    else
        mkdir -p "$(dirname "$selection_path")"
        python "$SELECTION_BUILDER" "${selection_flags[@]}" --output "$selection_path"
    fi

    extraction_export="ALL,PROJECT_ROOT=$PROJECT_ROOT,CHECKPOINT_DIR=$checkpoint,CACHE_DIR=$cache_dir,CONDITION=$condition,SKIP_CLASSIFIERS=1,SUBJECT_SELECTION=$selection_path,ENV_ACTIVATE=$ENV_ACTIVATE,LOG_ROOT=$log_root,QWEN_HIDDEN_DEPS=$QWEN_HIDDEN_DEPS"
    if [ -n "$MODEL_PATH" ]; then
        extraction_export="$extraction_export,MODEL_PATH=$MODEL_PATH"
    fi
    extraction_cmd=(sbatch --parsable --job-name="q3h-$(printf '%s' "$name" | cut -c1-24)" --gres="gpu:$gpus" --chdir="$PROJECT_ROOT" --export="$extraction_export" "$HIDDEN_WORKER")
    extraction_raw="$(submit "${extraction_cmd[@]}")"
    extraction_job="$(job_id "$extraction_raw")"

    classifier_export="ALL,PROJECT_ROOT=$PROJECT_ROOT,CACHE_DIR=$cache_dir,CLASSIFIER_DIR=$classifier_dir,CLASSIFIER_VARIANTS=$CLASSIFIER_VARIANTS,LOG_ROOT=$log_root,QWEN_HIDDEN_DEPS=$QWEN_HIDDEN_DEPS"
    classifier_cmd=(sbatch --parsable --job-name="q3hc-$(printf '%s' "$name" | cut -c1-23)" --dependency="afterok:$extraction_job" --chdir="$PROJECT_ROOT" --export="$classifier_export" "$CLASSIFIER_WORKER")
    classifier_raw="$(submit "${classifier_cmd[@]}")"
    classifier_job="$(job_id "$classifier_raw")"

    if [ "$DRY_RUN" = 0 ]; then
        printf '%s\textraction\t%s\t-\t%s\t%s\t%s\n' "$name" "$extraction_job" "$checkpoint" "$cache_dir" "$classifier_dir" >> "$registry"
        printf '%s\tclassifier\t%s\tafterok:%s\t%s\t%s\t%s\n' "$name" "$classifier_job" "$extraction_job" "$checkpoint" "$cache_dir" "$classifier_dir" >> "$registry"
        echo "submitted: $name extraction=$extraction_job classifier=$classifier_job"
    fi
done

echo "Qwen3 hidden-extraction smoke plan complete: jobs=$(( ${#TASKS[@]} * 2 )) dry_run=$DRY_RUN"
[ "$DRY_RUN" = 1 ] || echo "Smoke registry: $registry"
