#!/usr/bin/env bash
# Submit the 63-fold harmonized standalone matrix with maximum parallelism.
# One four-GPU train lane per training task plus one one-GPU auxiliary lane
# per aux task, so every cell starts as soon as the scheduler grants GPUs.
# There is no project-wide GPU cap.
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/gpfs/projects/etur92/ozu647717/AudioLLM/LLM-Depression}"
MATRIX="${MATRIX:-$PROJECT_ROOT/configs/experiments/harmonized/standalone_matrix.yaml}"
RUN_ID="${RUN_ID:?Set a unique RUN_ID}"
DRY_RUN="${DRY_RUN:-1}"
MAX_CONCURRENT_TRAINS="${MAX_CONCURRENT_TRAINS:-63}"
MAX_CONCURRENT_AUX="${MAX_CONCURRENT_AUX:-63}"
PREFLIGHT_AUDIT="${PREFLIGHT_AUDIT:-$PROJECT_ROOT/outputs/harmonized_mn5_preflight/$RUN_ID/audit.json}"
PREFLIGHT_COMPONENTS="${PREFLIGHT_COMPONENTS:-5}"
PREFLIGHT_MERGED="${PREFLIGHT_MERGED:-3}"
TRAIN_WORKER="${TRAIN_WORKER:-$PROJECT_ROOT/scripts/run_train_slurm.sh}"
EVAL_WORKER="${EVAL_WORKER:-$PROJECT_ROOT/scripts/run_eval_slurm.sh}"
HIDDEN_WORKER="${HIDDEN_WORKER:-$PROJECT_ROOT/scripts/run_qwen_hidden_extract_slurm.sh}"
GITHUB_ISSUE="${GITHUB_ISSUE:?Set the harmonized campaign GITHUB_ISSUE}"
GITHUB_PR="${GITHUB_PR:?Set the primary harmonized methodology GITHUB_PR}"
# Campaign-family defaults. The Qwen harmonized matrix keeps today's paths and
# naming; a Gemma matrix selects the Gemma output/context/feature roots and
# gemma4_* naming. Every knob remains overridable from the environment.
case "$MATRIX" in
  *gemma4*)
    SUBMISSIONS_ROOT="${SUBMISSIONS_ROOT:-$PROJECT_ROOT/outputs/gemma4_harmonized_submissions}"
    CONTEXTS_ROOT="${CONTEXTS_ROOT:-$PROJECT_ROOT/outputs/gemma4_experiment_contexts}"
    FEATURES_ROOT="${FEATURES_ROOT:-$PROJECT_ROOT/outputs/hidden_features/harmonized_v1_gemma4}"
    CLASSIFIERS_ROOT="${CLASSIFIERS_ROOT:-$PROJECT_ROOT/outputs/hidden_classifiers/harmonized_v1_gemma4}"
    RUN_PREFIX="${RUN_PREFIX:-gemma4_harmonized_v1}"
    GROUP_PREFIX="${GROUP_PREFIX:-gemma4-harmonized-v1}"
    LOGICAL_PREFIX="${LOGICAL_PREFIX:-gemma4_harmonized_v1}"
    EXPECTED_TRAIN_TASKS="${EXPECTED_TRAIN_TASKS:-60}"
    EXPECTED_EVAL_TASKS="${EXPECTED_EVAL_TASKS:-30}"
    ;;
  *)
    SUBMISSIONS_ROOT="${SUBMISSIONS_ROOT:-$PROJECT_ROOT/outputs/harmonized_submissions}"
    CONTEXTS_ROOT="${CONTEXTS_ROOT:-$PROJECT_ROOT/outputs/harmonized_experiment_contexts}"
    FEATURES_ROOT="${FEATURES_ROOT:-$PROJECT_ROOT/outputs/hidden_features/harmonized_v1}"
    CLASSIFIERS_ROOT="${CLASSIFIERS_ROOT:-$PROJECT_ROOT/outputs/hidden_classifiers/harmonized_v1}"
    RUN_PREFIX="${RUN_PREFIX:-harmonized_v1}"
    GROUP_PREFIX="${GROUP_PREFIX:-harmonized-v1}"
    LOGICAL_PREFIX="${LOGICAL_PREFIX:-harmonized_v1}"
    EXPECTED_TRAIN_TASKS="${EXPECTED_TRAIN_TASKS:-63}"
    EXPECTED_EVAL_TASKS="${EXPECTED_EVAL_TASKS:-33}"
    ;;
esac

case "$DRY_RUN" in 0|1) ;; *) echo "DRY_RUN must be 0 or 1" >&2; exit 2;; esac
case "$GITHUB_ISSUE" in ''|*[!0-9]*|0) echo "GITHUB_ISSUE must be a positive integer." >&2; exit 2;; esac
case "$GITHUB_PR" in ''|*[!0-9]*|0) echo "GITHUB_PR must be a positive integer." >&2; exit 2;; esac
if [ "$MAX_CONCURRENT_TRAINS" -lt 1 ] || [ "$MAX_CONCURRENT_AUX" -lt 1 ]; then
    echo "Concurrency limits must be positive." >&2
    exit 2
fi
for path in "$MATRIX" "$TRAIN_WORKER" "$EVAL_WORKER" "$HIDDEN_WORKER"; do
    [ -f "$path" ] || { echo "Missing required file: $path" >&2; exit 3; }
done

if [ "$DRY_RUN" = 0 ]; then
    python - "$PREFLIGHT_AUDIT" "$RUN_ID" "$PREFLIGHT_COMPONENTS" "$PREFLIGHT_MERGED" <<'PY'
import json, sys
payload = json.load(open(sys.argv[1], encoding="utf-8"))
if payload.get("status") != "passed" or payload.get("run_id") != sys.argv[2]:
    raise SystemExit(f"Incompatible MN5 preflight audit: {sys.argv[1]}")
components = payload.get("components", [])
merged = payload.get("merged")
if len(components) != int(sys.argv[3]):
    raise SystemExit(f"Incomplete MN5 preflight audit: {sys.argv[1]}")
if int(sys.argv[4]) > 0 and len(merged or []) != int(sys.argv[4]):
    raise SystemExit(f"Incomplete MN5 preflight audit: {sys.argv[1]}")
if int(sys.argv[4]) == 0 and merged not in (None, [], {}):
    raise SystemExit(f"Preflight audit must not contain merged records: {sys.argv[1]}")
PY
fi

cd "$PROJECT_ROOT"
mapfile -t TASKS < <(python - "$MATRIX" "$PROJECT_ROOT" "$EXPECTED_TRAIN_TASKS" "$EXPECTED_EVAL_TASKS" <<'PY'
import sys, yaml
from pathlib import Path
root = Path(sys.argv[2])
expected_train = int(sys.argv[3])
expected_eval = int(sys.argv[4])
matrix = yaml.safe_load(Path(sys.argv[1]).read_text(encoding="utf-8"))
backend = str(matrix.get("model_backend", "") or "")
expected_heads = ["logreg_raw"] if backend == "gemma4" else ["logreg_raw", "xgb_raw"]
if matrix.get("fixed_heads") != expected_heads:
    raise SystemExit(
        f"Harmonized matrix must contain only {expected_heads} heads for "
        f"model_backend={backend or 'qwen'}"
    )
count = 0
eval_count = 0
for item in matrix["experiments"]:
    config_path = root / item["config"]
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if int(config["training"]["num_train_epochs"]) != 20:
        raise SystemExit(f"Expected 20 epochs: {config_path}")
    if config["training"]["selection_metric"] != "inner_val_macro_f1":
        raise SystemExit(f"Expected macro-F1 selection: {config_path}")
    config_backend = str(config.get("model_backend", "") or "")
    if config_backend != backend:
        raise SystemExit(
            f"Matrix model_backend={backend!r} does not match config backend "
            f"{config_backend!r}: {config_path}"
        )
    modality = "audio_text" if config["data"].get("use_audio") and config["data"].get("use_text") else "audio_only" if config["data"].get("use_audio") else "text_only"
    run_root = str(config["output_dirs"]["run_root"]).replace("${PROJECT_ROOT}", str(root))
    for fold in item["folds"]:
        print("\t".join((str(config_path), str(config["dataset"]), modality, str(fold), "1" if item["separate_eval"] else "0", run_root, config_backend)))
        count += 1
        if item["separate_eval"]:
            eval_count += 1
if count != expected_train or eval_count != expected_eval:
    raise SystemExit(
        f"Expected {expected_train} standalone fold tasks with "
        f"{expected_eval} separate evals, found {count}/{eval_count}"
    )
PY
)

submit() {
    if [ "$DRY_RUN" = 1 ]; then
        printf 'DRY_RUN ' >&2; printf '%q ' "$@" >&2; printf '\n' >&2
        printf 'dry_%s\n' "$(printf '%s\0' "$@" | sha256sum | cut -c1-12)"
    else
        "$@"
    fi
}
job_id() { printf '%s' "${1%%;*}"; }
dependency_arg() {
    local chain="${1:-}" throttle="${2:-}" value=""
    if [ -n "$chain" ]; then value="afterok:$chain"; fi
    if [ -n "$throttle" ] && [ "$throttle" != "$chain" ]; then
        value="${value:+$value,}afterany:$throttle"
    fi
    [ -n "$value" ] && printf '%s' "--dependency=$value"
}

train_index=0
aux_index=0
declare -a train_lanes aux_lanes
registry="$SUBMISSIONS_ROOT/$RUN_ID/jobs.tsv"
if [ "$DRY_RUN" = 0 ]; then
    [ ! -e "$registry" ] || { echo "Refusing existing submission registry: $registry" >&2; exit 4; }
    mkdir -p "$(dirname "$registry")"
    printf 'dataset\tmodality\tfold\tkind\tjob_id\tdependency\n' > "$registry"
fi

for task in "${TASKS[@]}"; do
    IFS=$'\t' read -r config dataset modality fold separate_eval run_root config_backend <<< "$task"
    backend_vars="$(bash "$PROJECT_ROOT/scripts/harmonized_backend_env.sh" "$config" "$PROJECT_ROOT")"
    eval "$backend_vars"
    run_name="${RUN_PREFIX}_${RUN_ID}_${dataset}_${modality}"
    fold_dir="$run_root/$run_name/fold_$fold"
    context_dir="$CONTEXTS_ROOT/$RUN_ID/$dataset/$modality/fold_$fold"
    context_path="$context_dir/context.json"
    if [ "$DRY_RUN" = 0 ]; then
        mkdir -p "$context_dir"
        python - "$context_path" "$PREFLIGHT_AUDIT" "$RUN_ID" "$dataset" "$modality" "$fold" "$run_name" "$PROJECT_ROOT" "$GITHUB_ISSUE" "$GITHUB_PR" "$GROUP_PREFIX" "$LOGICAL_PREFIX" "$config_backend" <<'PY'
import json, sys
from pathlib import Path
sys.path.insert(0, sys.argv[12])
from src.experiment_tracking.identity import new_attempt_id
context_path, audit_path, run_id, dataset, modality, fold, run_name, root, github_issue, github_pr, group_prefix, logical_prefix, config_backend = sys.argv[1:]
audit = json.load(open(audit_path, encoding="utf-8"))
component = next(item for item in audit["components"] if item["dataset"] == dataset)
commit = str(audit.get("source_commit") or "")
if len(commit) != 40:
    raise SystemExit("Preflight audit has no full source commit")
logical = f"{logical_prefix}_{dataset}_{modality}_seed1337"
payload = {
    "schema_version": "audiollm.experiment_context.v1",
    "group_id": f"{group_prefix}-{run_id}",
    "logical_run_name": logical,
    "attempt_id": new_attempt_id(logical, commit),
    "fold": int(fold),
    "seed": 1337,
    "source": {"git_commit": commit, "git_branch": audit.get("source_branch"), "git_dirty": False},
    "research": {"github_issue": int(github_issue), "github_pr": int(github_pr)},
    "hashes": {"manifest_sha256": component["manifest_file_sha256"], "split_sha256": component["split_metadata_sha256"]},
    "model_backend": config_backend or None,
    "slurm": {"train_job_id": None, "eval_job_ids": []},
}
path = Path(context_path)
if path.exists():
    raise SystemExit(f"Refusing existing experiment context: {path}")
path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY
    fi

    train_lane=$((train_index % MAX_CONCURRENT_TRAINS))
    train_throttle="${train_lanes[$train_lane]:-}"
    train_dep="$(dependency_arg "" "$train_throttle" || true)"
    export_spec="ALL,PROJECT_ROOT=$PROJECT_ROOT,CONFIG=$config,FOLD=$fold,RUN_NAME=$run_name,SKIP_MANIFEST_BUILD=1,EXPERIMENT_CONTEXT=$context_path,ENV_ACTIVATE=$ENV_ACTIVATE,MODEL_PATH=$MODEL_PATH"
    train_cmd=(sbatch --parsable --job-name="h-${dataset:0:4}-${modality:0:2}-f$fold" --export="$export_spec")
    [ -n "$train_dep" ] && train_cmd+=("$train_dep")
    train_cmd+=("$TRAIN_WORKER")
    train_raw="$(submit "${train_cmd[@]}")"
    train_job="$(job_id "$train_raw")"
    train_lanes[$train_lane]="$train_job"
    train_index=$((train_index + 1))
    [ "$DRY_RUN" = 1 ] || printf '%s\t%s\t%s\ttrain\t%s\t%s\n' "$dataset" "$modality" "$fold" "$train_job" "$train_throttle" >> "$registry"

    chain_job="$train_job"
    if [ "$separate_eval" = 1 ]; then
        aux_lane=$((aux_index % MAX_CONCURRENT_AUX))
        aux_throttle="${aux_lanes[$aux_lane]:-}"
        eval_dep="$(dependency_arg "$train_job" "$aux_throttle")"
        eval_cmd=(sbatch --parsable --job-name="he-${dataset:0:4}-${modality:0:2}-f$fold" "$eval_dep" --export="$export_spec,CHECKPOINT_DIR=$fold_dir/best_model,OUTPUT_DIR=$fold_dir/best_model/standalone_eval" "$EVAL_WORKER")
        eval_raw="$(submit "${eval_cmd[@]}")"
        chain_job="$(job_id "$eval_raw")"
        aux_lanes[$aux_lane]="$chain_job"
        aux_index=$((aux_index + 1))
        [ "$DRY_RUN" = 1 ] || printf '%s\t%s\t%s\teval\t%s\t%s,%s\n' "$dataset" "$modality" "$fold" "$chain_job" "$train_job" "$aux_throttle" >> "$registry"
    fi

    aux_lane=$((aux_index % MAX_CONCURRENT_AUX))
    aux_throttle="${aux_lanes[$aux_lane]:-}"
    hidden_dep="$(dependency_arg "$chain_job" "$aux_throttle")"
    cache="$FEATURES_ROOT/$dataset/$run_name/fold_$fold"
    classifiers="$CLASSIFIERS_ROOT/$dataset/$run_name/fold_$fold"
    hidden_cmd=(sbatch --parsable --job-name="hh-${dataset:0:4}-${modality:0:2}-f$fold" "$hidden_dep" --export="ALL,PROJECT_ROOT=$PROJECT_ROOT,CHECKPOINT_DIR=$fold_dir/best_model,CACHE_DIR=$cache,CLASSIFIER_DIR=$classifiers,MODEL_PATH=$MODEL_PATH,CONDITION=$modality,CLASSIFIER_VARIANTS=$CLASSIFIER_VARIANTS" "$HIDDEN_WORKER")
    hidden_raw="$(submit "${hidden_cmd[@]}")"
    hidden_job="$(job_id "$hidden_raw")"
    aux_lanes[$aux_lane]="$hidden_job"
    aux_index=$((aux_index + 1))
    [ "$DRY_RUN" = 1 ] || printf '%s\t%s\t%s\thidden_fixed\t%s\t%s,%s\n' "$dataset" "$modality" "$fold" "$hidden_job" "$chain_job" "$aux_throttle" >> "$registry"
    if [ "$DRY_RUN" = 0 ]; then
        python - "$context_path" "$fold_dir" "$train_job" "${eval_raw:-}" "$hidden_job" "$PROJECT_ROOT" <<'PY'
import json, os, sys
from pathlib import Path
sys.path.insert(0, sys.argv[6])
from src.experiment_tracking import lifecycle

context_path, fold_dir, train_job, eval_raw, hidden_job = sys.argv[1:6]
eval_job = eval_raw.split(";", 1)[0] if eval_raw else None
path = Path(context_path)
context = json.loads(path.read_text(encoding="utf-8"))
context["slurm"] = {"train_job_id": train_job, "eval_job_ids": [eval_job] if eval_job else []}
temporary = path.with_suffix(".tmp")
temporary.write_text(json.dumps(context, indent=2) + "\n", encoding="utf-8")
os.replace(temporary, path)

run_root = Path(fold_dir)
run_root.mkdir(parents=True, exist_ok=True)
events = [
    lifecycle.new_job_event(
        job_key="train", job_type="train", event_type="SUBMITTED",
        attempt_id=context["attempt_id"], fold=int(context["fold"]),
        slurm_job_id=train_job, status="PENDING",
    )
]
if eval_job:
    events.append(lifecycle.new_job_event(
        job_key="best_eval", job_type="evaluation", event_type="SUBMITTED",
        attempt_id=context["attempt_id"], fold=int(context["fold"]),
        slurm_job_id=eval_job, dependency_job_ids=[train_job], status="PENDING",
    ))
events.append(lifecycle.new_job_event(
    job_key="hidden_fixed", job_type="hidden_classifier", event_type="SUBMITTED",
    attempt_id=context["attempt_id"], fold=int(context["fold"]),
    slurm_job_id=hidden_job, dependency_job_ids=[eval_job or train_job], status="PENDING",
))
for event in events:
    lifecycle.append_job_event(run_root / "jobs.jsonl", event)
PY
    fi
    unset eval_raw
done

echo "Harmonized standalone plan: tasks=${#TASKS[@]} train_lanes=$MAX_CONCURRENT_TRAINS aux_lanes=$MAX_CONCURRENT_AUX max_gpus=$((MAX_CONCURRENT_TRAINS * 4 + MAX_CONCURRENT_AUX)) github_issue=$GITHUB_ISSUE github_pr=$GITHUB_PR dry_run=$DRY_RUN"
[ "$DRY_RUN" = 1 ] || echo "Submission registry: $registry"
