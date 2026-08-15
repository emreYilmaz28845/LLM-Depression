#!/usr/bin/env bash
# Targeted, tracking-safe retries for failed harmonized standalone cells.
# The main launcher refuses an existing jobs.tsv and resubmits the whole
# matrix, so failed cells are retried here with new attempt identities,
# new run names (or new output directories), resubmission chains, and a
# separate registry. The original jobs.tsv, contexts, sidecars, and logs
# are never modified except for appending terminal events (FAILED or
# CANCELLED, see ORIGINAL_TERMINAL_EVENT) for the original attempt's jobs
# and the new attempt's SUBMITTED events to the fold's append-only
# jobs.jsonl.
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/gpfs/projects/etur92/ozu647717/AudioLLM/LLM-Depression}"
RUN_ID="${RUN_ID:?Set the shared harmonized RUN_ID}"
CELLS="${CELLS:?Set CELLS to a TSV of failed cells}"
DRY_RUN="${DRY_RUN:-1}"
RETRY_TAG="${RETRY_TAG:-r1}"
REASON="${REASON:-retry of a failed harmonized standalone cell}"
ORIGINAL_TERMINAL_EVENT="${ORIGINAL_TERMINAL_EVENT:-FAILED}"
case "$ORIGINAL_TERMINAL_EVENT" in FAILED|CANCELLED) ;; *) echo "ORIGINAL_TERMINAL_EVENT must be FAILED or CANCELLED" >&2; exit 2;; esac
MAX_CONCURRENT_TRAINS="${MAX_CONCURRENT_TRAINS:-15}"
MAX_CONCURRENT_AUX="${MAX_CONCURRENT_AUX:-4}"
GPU_CEILING=64
PREFLIGHT_AUDIT="${PREFLIGHT_AUDIT:-$PROJECT_ROOT/outputs/harmonized_mn5_preflight/$RUN_ID/audit.json}"
PREFLIGHT_COMPONENTS="${PREFLIGHT_COMPONENTS:-5}"
PREFLIGHT_MERGED="${PREFLIGHT_MERGED:-3}"
TRAIN_WORKER="${TRAIN_WORKER:-$PROJECT_ROOT/scripts/run_train_slurm.sh}"
EVAL_WORKER="${EVAL_WORKER:-$PROJECT_ROOT/scripts/run_eval_slurm.sh}"
HIDDEN_WORKER="${HIDDEN_WORKER:-$PROJECT_ROOT/scripts/run_qwen_hidden_extract_slurm.sh}"
GITHUB_ISSUE="${GITHUB_ISSUE:?Set the harmonized campaign GITHUB_ISSUE}"
GITHUB_PR="${GITHUB_PR:?Set the primary harmonized methodology GITHUB_PR}"
MATRIX="${MATRIX:-$PROJECT_ROOT/configs/experiments/harmonized/standalone_matrix.yaml}"
# Campaign-family defaults. A Gemma matrix selects the Gemma roots and
# naming; the Qwen matrix keeps today's values. The case must run before the
# defaults are applied so ${VAR:-default} cannot lock the Qwen value first.
case "$MATRIX" in
  *gemma4*)
    SUBMISSIONS_ROOT="${SUBMISSIONS_ROOT:-$PROJECT_ROOT/outputs/gemma4_harmonized_submissions}"
    CONTEXTS_ROOT="${CONTEXTS_ROOT:-$PROJECT_ROOT/outputs/gemma4_experiment_contexts}"
    FEATURES_ROOT="${FEATURES_ROOT:-$PROJECT_ROOT/outputs/hidden_features/harmonized_v1_gemma4}"
    CLASSIFIERS_ROOT="${CLASSIFIERS_ROOT:-$PROJECT_ROOT/outputs/hidden_classifiers/harmonized_v1_gemma4}"
    RUN_PREFIX="${RUN_PREFIX:-gemma4_harmonized_v1}"
    GROUP_PREFIX="${GROUP_PREFIX:-gemma4-harmonized-v1}"
    LOGICAL_PREFIX="${LOGICAL_PREFIX:-gemma4_harmonized_v1}"
    ;;
  *)
    SUBMISSIONS_ROOT="${SUBMISSIONS_ROOT:-$PROJECT_ROOT/outputs/harmonized_submissions}"
    CONTEXTS_ROOT="${CONTEXTS_ROOT:-$PROJECT_ROOT/outputs/harmonized_experiment_contexts}"
    FEATURES_ROOT="${FEATURES_ROOT:-$PROJECT_ROOT/outputs/hidden_features/harmonized_v1}"
    CLASSIFIERS_ROOT="${CLASSIFIERS_ROOT:-$PROJECT_ROOT/outputs/hidden_classifiers/harmonized_v1}"
    RUN_PREFIX="${RUN_PREFIX:-harmonized_v1}"
    GROUP_PREFIX="${GROUP_PREFIX:-harmonized-v1}"
    LOGICAL_PREFIX="${LOGICAL_PREFIX:-harmonized_v1}"
    ;;
esac

case "$DRY_RUN" in 0|1) ;; *) echo "DRY_RUN must be 0 or 1" >&2; exit 2;; esac
case "$GITHUB_ISSUE" in ''|*[!0-9]*|0) echo "GITHUB_ISSUE must be a positive integer." >&2; exit 2;; esac
case "$GITHUB_PR" in ''|*[!0-9]*|0) echo "GITHUB_PR must be a positive integer." >&2; exit 2;; esac
if [ $((MAX_CONCURRENT_TRAINS * 4 + MAX_CONCURRENT_AUX)) -gt "$GPU_CEILING" ]; then
    echo "Requested lanes can exceed the $GPU_CEILING-GPU project ceiling: trains=$MAX_CONCURRENT_TRAINS aux=$MAX_CONCURRENT_AUX" >&2
    exit 2
fi
for path in "$CELLS" "$MATRIX" "$TRAIN_WORKER" "$EVAL_WORKER" "$HIDDEN_WORKER"; do
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
# Resolve the matrix so a cell can find its config path, run root, and
# separate-eval flag. A cell that names an unknown dataset/modality stops.
mapfile -t CELL_SPECS < <(python - "$MATRIX" "$PROJECT_ROOT" "$CELLS" <<'PY'
import sys, yaml
from pathlib import Path
root = Path(sys.argv[2])
matrix = yaml.safe_load(Path(sys.argv[1]).read_text(encoding="utf-8"))
cells = []
for line in Path(sys.argv[3]).read_text(encoding="utf-8").splitlines():
    line = line.strip()
    if not line or line.startswith("#") or line.startswith("dataset"):
        continue
    parts = line.split("\t")
    if len(parts) < 7:
        raise SystemExit(f"Cells row must have at least 7 columns (dataset modality fold train_ok failed_train failed_eval failed_hidden): {line}")
    for term in parts[7:10]:
        if term not in ("", "FAILED", "CANCELLED"):
            raise SystemExit(f"Terminal-event columns must be FAILED, CANCELLED, or empty: {line}")
    if parts[3] not in ("0", "1"):
        raise SystemExit(f"train_ok must be 0 or 1: {line}")
    try:
        int(parts[2])
    except ValueError:
        raise SystemExit(f"fold must be an integer: {line}")
    cells.append(parts)
wanted = {(c[0], c[1]) for c in cells}
matched = set()
for item in matrix["experiments"]:
    config_path = root / item["config"]
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    modality = "audio_text" if config["data"].get("use_audio") and config["data"].get("use_text") else "audio_only" if config["data"].get("use_audio") else "text_only"
    key = (str(config["dataset"]), modality)
    if key not in wanted:
        continue
    matched.add(key)
    run_root = str(config["output_dirs"]["run_root"]).replace("${PROJECT_ROOT}", str(root))
    for cell in cells:
        if (cell[0], cell[1]) == key:
            terms = [cell[i] if len(cell) > i else "" for i in (7, 8, 9)]
            print("\t".join((str(config_path), run_root, "1" if item["separate_eval"] else "0", cell[0], cell[1], cell[2], cell[3], cell[4], cell[5], cell[6] if len(cell) > 6 else "", *terms)))
missing = wanted - matched
if missing:
    raise SystemExit(f"Cells reference unknown dataset/modality: {sorted(missing)}")
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
registry="$SUBMISSIONS_ROOT/$RUN_ID/retry_${RETRY_TAG}_jobs.tsv"
if [ "$DRY_RUN" = 0 ]; then
    [ ! -e "$registry" ] || { echo "Refusing existing retry registry: $registry" >&2; exit 4; }
    mkdir -p "$(dirname "$registry")"
    printf 'dataset\tmodality\tfold\tkind\tjob_id\tdependency\tresubmission_of\n' > "$registry"
fi

for spec in "${CELL_SPECS[@]}"; do
    # bash read with tab IFS treats tab as whitespace and collapses
    # consecutive empty fields, shifting empty failed_eval/failed_train
    # columns into the wrong variables. Parse the 13 fields explicitly.
    eval "$(python3 - "$spec" <<'PY'
import shlex, sys
fields = sys.argv[1].split("\t")
names = [
    "config", "run_root", "separate_eval", "dataset", "modality", "fold",
    "train_ok", "failed_train", "failed_eval", "failed_hidden",
    "train_term", "eval_term", "hidden_term",
]
for index, name in enumerate(names):
    print(f"{name}={shlex.quote(fields[index] if index < len(fields) else '')}")
PY
)"
    backend_vars="$(bash "$PROJECT_ROOT/scripts/harmonized_backend_env.sh" "$config" "$PROJECT_ROOT")"
    eval "$backend_vars"
    run_name_base="${RUN_PREFIX}_${RUN_ID}_${dataset}_${modality}"
    if [ "$train_ok" = "1" ]; then
        run_name="$run_name_base"
        fold_dir="$run_root/$run_name/fold_$fold"
        context_dir="$CONTEXTS_ROOT/$RUN_ID/retry_${RETRY_TAG}/$dataset/$modality/fold_$fold"
    else
        run_name="${run_name_base}_${RETRY_TAG}"
        fold_dir="$run_root/$run_name/fold_$fold"
        context_dir="$CONTEXTS_ROOT/$RUN_ID/retry_${RETRY_TAG}/$dataset/$modality/fold_$fold"
    fi
    context_path="$context_dir/context.json"
    original_fold_dir="$run_root/${run_name_base}/fold_$fold"
    original_context_path="$CONTEXTS_ROOT/$RUN_ID/$dataset/$modality/fold_$fold/context.json"

    if [ "$DRY_RUN" = 0 ]; then
        [ -f "$original_context_path" ] || { echo "Missing original experiment context: $original_context_path" >&2; exit 3; }
        mkdir -p "$context_dir"
        python - "$context_path" "$PREFLIGHT_AUDIT" "$RUN_ID" "$dataset" "$modality" "$fold" "$run_name" "$PROJECT_ROOT" "$GITHUB_ISSUE" "$GITHUB_PR" "$RETRY_TAG" "$GROUP_PREFIX" "$LOGICAL_PREFIX" <<'PY'
import json, sys
from pathlib import Path
sys.path.insert(0, sys.argv[9])
from src.experiment_tracking.identity import new_attempt_id
context_path, audit_path, run_id, dataset, modality, fold, run_name, root, github_issue, github_pr, retry_tag, group_prefix, logical_prefix = sys.argv[1:]
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
    "slurm": {"train_job_id": None, "eval_job_ids": []},
    "retry_tag": retry_tag,
}
path = Path(context_path)
if path.exists():
    raise SystemExit(f"Refusing existing experiment context: {path}")
path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY
    fi

    chain_job=""
    if [ "$train_ok" = "0" ]; then
        train_lane=$((train_index % MAX_CONCURRENT_TRAINS))
        train_throttle="${train_lanes[$train_lane]:-}"
        train_dep="$(dependency_arg "" "$train_throttle" || true)"
        export_spec="ALL,PROJECT_ROOT=$PROJECT_ROOT,CONFIG=$config,FOLD=$fold,RUN_NAME=$run_name,SKIP_MANIFEST_BUILD=1,EXPERIMENT_CONTEXT=$context_path,ENV_ACTIVATE=$ENV_ACTIVATE,MODEL_PATH=$MODEL_PATH"
        train_cmd=(sbatch --parsable --job-name="hr-${dataset:0:4}-${modality:0:2}-f$fold" --export="$export_spec")
        [ -n "$train_dep" ] && train_cmd+=("$train_dep")
        train_cmd+=("$TRAIN_WORKER")
        train_raw="$(submit "${train_cmd[@]}")"
        chain_job="$(job_id "$train_raw")"
        train_lanes[$train_lane]="$chain_job"
        train_index=$((train_index + 1))
        [ "$DRY_RUN" = 1 ] || printf '%s\t%s\t%s\ttrain\t%s\t%s\t%s\n' "$dataset" "$modality" "$fold" "$chain_job" "$train_throttle" "$failed_train" >> "$registry"
    else
        chain_job="$failed_train"
    fi

    if [ "$separate_eval" = "1" ]; then
        aux_lane=$((aux_index % MAX_CONCURRENT_AUX))
        aux_throttle="${aux_lanes[$aux_lane]:-}"
        if [ "$train_ok" = "1" ]; then
            # The original train already completed; its job id may have been
            # purged from Slurm accounting, so no afterok dependency is used.
            eval_dep="$(dependency_arg "" "$aux_throttle" || true)"
            eval_chain=""
            eval_out="$fold_dir/best_model/standalone_eval_${RETRY_TAG}"
        else
            eval_dep="$(dependency_arg "$chain_job" "$aux_throttle")"
            eval_chain="$chain_job"
            eval_out="$fold_dir/best_model/standalone_eval"
        fi
        eval_cmd=(sbatch --parsable --job-name="hre-${dataset:0:4}-${modality:0:2}-f$fold")
        [ -n "$eval_dep" ] && eval_cmd+=("$eval_dep")
        eval_cmd+=(--export="ALL,PROJECT_ROOT=$PROJECT_ROOT,CONFIG=$config,FOLD=$fold,RUN_NAME=$run_name,SKIP_MANIFEST_BUILD=1,EXPERIMENT_CONTEXT=$context_path,ENV_ACTIVATE=$ENV_ACTIVATE,MODEL_PATH=$MODEL_PATH,CHECKPOINT_DIR=$fold_dir/best_model,OUTPUT_DIR=$eval_out" "$EVAL_WORKER")
        eval_raw="$(submit "${eval_cmd[@]}")"
        eval_job="$(job_id "$eval_raw")"
        aux_lanes[$aux_lane]="$eval_job"
        aux_index=$((aux_index + 1))
        [ "$DRY_RUN" = 1 ] || printf '%s\t%s\t%s\teval\t%s\t%s,%s\t%s\n' "$dataset" "$modality" "$fold" "$eval_job" "$eval_chain" "$aux_throttle" "$failed_eval" >> "$registry"
        chain_job="$eval_job"
    fi

    aux_lane=$((aux_index % MAX_CONCURRENT_AUX))
    aux_throttle="${aux_lanes[$aux_lane]:-}"
    hidden_dep="$(dependency_arg "$chain_job" "$aux_throttle")"
    if [ "$train_ok" = "1" ]; then
        cache="$FEATURES_ROOT/$dataset/${run_name_base}_${RETRY_TAG}/fold_$fold"
        classifiers="$CLASSIFIERS_ROOT/$dataset/${run_name_base}_${RETRY_TAG}/fold_$fold"
    else
        cache="$FEATURES_ROOT/$dataset/$run_name/fold_$fold"
        classifiers="$CLASSIFIERS_ROOT/$dataset/$run_name/fold_$fold"
    fi
    hidden_cmd=(sbatch --parsable --job-name="hrh-${dataset:0:4}-${modality:0:2}-f$fold" "$hidden_dep" --export="ALL,PROJECT_ROOT=$PROJECT_ROOT,CHECKPOINT_DIR=$fold_dir/best_model,CACHE_DIR=$cache,CLASSIFIER_DIR=$classifiers,MODEL_PATH=$MODEL_PATH,CONDITION=$modality,CLASSIFIER_VARIANTS=$CLASSIFIER_VARIANTS" "$HIDDEN_WORKER")
    hidden_raw="$(submit "${hidden_cmd[@]}")"
    hidden_job="$(job_id "$hidden_raw")"
    aux_lanes[$aux_lane]="$hidden_job"
    aux_index=$((aux_index + 1))
    [ "$DRY_RUN" = 1 ] || printf '%s\t%s\t%s\thidden_fixed\t%s\t%s,%s\t%s\n' "$dataset" "$modality" "$fold" "$hidden_job" "$chain_job" "$aux_throttle" "$failed_hidden" >> "$registry"

    if [ "$DRY_RUN" = 0 ]; then
        python - "$context_path" "$fold_dir" "$chain_job" "${eval_raw:-}" "$hidden_job" "$train_ok" "$failed_train" "$failed_eval" "$failed_hidden" "$PROJECT_ROOT" <<'PY'
import json, os, sys
from pathlib import Path
sys.path.insert(0, sys.argv[10])
from src.experiment_tracking import lifecycle

context_path, fold_dir, chain_job, eval_raw, hidden_job, train_ok, failed_train, failed_eval, failed_hidden = sys.argv[1:10]
path = Path(context_path)
context = json.loads(path.read_text(encoding="utf-8"))
eval_job = eval_raw.split(";", 1)[0] if eval_raw else None
context["slurm"] = {"train_job_id": None if train_ok == "1" else chain_job, "eval_job_ids": [eval_job] if eval_job else []}
temporary = path.with_suffix(".tmp")
temporary.write_text(json.dumps(context, indent=2) + "\n", encoding="utf-8")
os.replace(temporary, path)

run_root = Path(fold_dir)
run_root.mkdir(parents=True, exist_ok=True)
attempt = context["attempt_id"]
fold_n = int(context["fold"])
events = []
if train_ok == "0":
    events.append(lifecycle.new_job_event(
        job_key="train", job_type="train", event_type="SUBMITTED",
        attempt_id=attempt, fold=fold_n,
        slurm_job_id=chain_job, status="PENDING",
        resubmission_of_job_id=failed_train or None,
    ))
if eval_job:
    events.append(lifecycle.new_job_event(
        job_key="best_eval", job_type="evaluation", event_type="SUBMITTED",
        attempt_id=attempt, fold=fold_n,
        slurm_job_id=eval_job,
        dependency_job_ids=[] if train_ok == "1" else [chain_job], status="PENDING",
        resubmission_of_job_id=failed_eval or None,
    ))
events.append(lifecycle.new_job_event(
    job_key="hidden_fixed", job_type="hidden_classifier", event_type="SUBMITTED",
    attempt_id=attempt, fold=fold_n,
    slurm_job_id=hidden_job, dependency_job_ids=[eval_job or chain_job], status="PENDING",
    resubmission_of_job_id=failed_hidden or None,
))
for event in events:
    lifecycle.append_job_event(run_root / "jobs.jsonl", event)
PY
        python - "$original_fold_dir" "$original_context_path" "$train_ok" "$failed_train" "$failed_eval" "$failed_hidden" "$REASON" "$ORIGINAL_TERMINAL_EVENT" "$train_term" "$eval_term" "$hidden_term" "$PROJECT_ROOT" <<'PY'
import json, sys
from pathlib import Path
sys.path.insert(0, sys.argv[12])
from src.experiment_tracking import lifecycle

original_fold_dir, original_context_path, train_ok, failed_train, failed_eval, failed_hidden, reason, terminal_event, train_term, eval_term, hidden_term = sys.argv[1:12]
context = json.loads(Path(original_context_path).read_text(encoding="utf-8"))
attempt = context["attempt_id"]
fold_n = int(context["fold"])

def terminal_type(value: str) -> str:
    return value if value in ("FAILED", "CANCELLED") else terminal_event

events = []
if train_ok == "0":
    event_type = terminal_type(train_term)
    events.append(lifecycle.new_job_event(
        job_key="train", job_type="train", event_type=event_type,
        attempt_id=attempt, fold=fold_n,
        slurm_job_id=failed_train or None, status=event_type, reason=reason,
    ))
if failed_eval and failed_eval not in ("", "None"):
    event_type = terminal_type(eval_term)
    events.append(lifecycle.new_job_event(
        job_key="best_eval", job_type="evaluation", event_type=event_type,
        attempt_id=attempt, fold=fold_n,
        slurm_job_id=failed_eval, status=event_type, reason=reason,
    ))
if failed_hidden and failed_hidden not in ("", "None"):
    event_type = terminal_type(hidden_term)
    events.append(lifecycle.new_job_event(
        job_key="hidden_fixed", job_type="hidden_classifier", event_type=event_type,
        attempt_id=attempt, fold=fold_n,
        slurm_job_id=failed_hidden, status=event_type, reason=reason,
    ))
for event in events:
    lifecycle.append_job_event(Path(original_fold_dir) / "jobs.jsonl", event)
PY
    fi
    unset eval_raw
done

echo "Harmonized standalone retry plan: cells=${#CELL_SPECS[@]} train_lanes=$MAX_CONCURRENT_TRAINS aux_lanes=$MAX_CONCURRENT_AUX max_gpus=$((MAX_CONCURRENT_TRAINS * 4 + MAX_CONCURRENT_AUX)) github_issue=$GITHUB_ISSUE github_pr=$GITHUB_PR dry_run=$DRY_RUN"
[ "$DRY_RUN" = 1 ] || echo "Retry registry: $registry"
