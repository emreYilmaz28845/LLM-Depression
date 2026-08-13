#!/usr/bin/env bash
# Targeted, tracking-safe retries for failed DAIC official-development cells.
#
# The main launcher refuses an existing submission registry and resubmits the
# whole campaign, so failed cells are retried here with new attempt
# identities, resubmission chains, and terminal events appended to the
# original attempts. A train failure retries the full cell with a new attempt
# (supersedes the old one). An eval failure resubmits eval + extract + heads
# under the same training attempt. An extract or heads failure resubmits the
# chain under a new fixed-head child attempt (supersedes any partial one).
# At most one retry per failed job is permitted for a recorded transient
# infrastructure failure.
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/gpfs/projects/etur92/ozu647717/AudioLLM/LLM-Depression}"
MATRIX="${MATRIX:-$PROJECT_ROOT/configs/experiments/daic_officialdev/matrix.yaml}"
RUN_ID="${RUN_ID:?Set the campaign RUN_ID}"
CELLS="${CELLS:?Set CELLS to a TSV of failed cells}"
DRY_RUN="${DRY_RUN:-1}"
RETRY_TAG="${RETRY_TAG:-r1}"
REASON="${REASON:-retry of a failed DAIC official-development cell}"
MAX_CONCURRENT_TRAINS="${MAX_CONCURRENT_TRAINS:-6}"
MAX_CONCURRENT_AUX="${MAX_CONCURRENT_AUX:-12}"
GPU_CEILING=64
PREFLIGHT_AUDIT="${PREFLIGHT_AUDIT:-$PROJECT_ROOT/outputs/daic_officialdev_mn5_preflight/$RUN_ID/audit.json}"
TRAIN_WORKER="${TRAIN_WORKER:-$PROJECT_ROOT/scripts/run_train_slurm.sh}"
EVAL_WORKER="${EVAL_WORKER:-$PROJECT_ROOT/scripts/run_eval_slurm.sh}"
EXTRACT_WORKER="${EXTRACT_WORKER:-$PROJECT_ROOT/scripts/run_daic_officialdev_extract_slurm.sh}"
HEADS_WORKER="${HEADS_WORKER:-$PROJECT_ROOT/scripts/run_daic_officialdev_heads_slurm.sh}"
GITHUB_ISSUE="${GITHUB_ISSUE:?Set the campaign GITHUB_ISSUE}"
GITHUB_PR="${GITHUB_PR:?Set the campaign GITHUB_PR}"
CONTEXTS_ROOT="${CONTEXTS_ROOT:-$PROJECT_ROOT/outputs/daic_officialdev_experiment_contexts}"
SUBMISSIONS_ROOT="${SUBMISSIONS_ROOT:-$PROJECT_ROOT/outputs/daic_officialdev_submissions}"
QWEN_ENV_ACTIVATE="${QWEN_ENV_ACTIVATE:-/gpfs/projects/etur92/ozu647717/venvs/qwen_mn5_rebuilt/bin/activate}"
GEMMA_ENV_ACTIVATE="${GEMMA_ENV_ACTIVATE:-/gpfs/projects/etur92/ozu647717/venvs/gemma4_12b_tf5_14_1/bin/activate}"
MODEL_PATH_QWEN="${MODEL_PATH_QWEN:-/gpfs/projects/etur92/ozu647717/models/Qwen2-Audio-7B-Instruct}"
MODEL_PATH_QWEN_TEXT="${MODEL_PATH_QWEN_TEXT:-/gpfs/projects/etur92/ozu647717/models/Qwen2-7B-Instruct}"
MODEL_PATH_GEMMA4="${MODEL_PATH_GEMMA4:-/gpfs/projects/etur92/ozu647717/models/gemma-4-12B-it/707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7}"

case "$DRY_RUN" in 0|1) ;; *) echo "DRY_RUN must be 0 or 1" >&2; exit 2;; esac
case "$GITHUB_ISSUE" in ''|*[!0-9]*|0) echo "GITHUB_ISSUE must be a positive integer." >&2; exit 2;; esac
case "$GITHUB_PR" in ''|*[!0-9]*|0) echo "GITHUB_PR must be a positive integer." >&2; exit 2;; esac
if [ $((MAX_CONCURRENT_TRAINS * 4 + MAX_CONCURRENT_AUX)) -gt "$GPU_CEILING" ]; then
    echo "Requested lanes can exceed the $GPU_CEILING-GPU project ceiling." >&2
    exit 2
fi
[ -f "$CELLS" ] || { echo "Missing cells file: $CELLS" >&2; exit 3; }

cd "$PROJECT_ROOT"
MERGE_SHA="$(tr -d '\n' < "$PROJECT_ROOT/.provenance/git_commit.txt" 2>/dev/null || git rev-parse HEAD)"
MERGE_BRANCH="$(tr -d '\n' < "$PROJECT_ROOT/.provenance/git_branch.txt" 2>/dev/null || git rev-parse --abbrev-ref HEAD)"
GROUP_ID="daic-officialdev-qwen-gemma-v1-${MERGE_SHA:0:12}"

RETRY_SPECS_RAW="$(python - "$CELLS" <<'PY'
import sys
rows = []
for line in open(sys.argv[1], encoding="utf-8"):
    line = line.strip()
    if not line or line.startswith("#") or line.startswith("backbone"):
        continue
    parts = line.split("\t")
    if len(parts) < 4:
        raise SystemExit(f"CELLS row needs at least 4 columns (backbone modality kind job_id ...): {line}")
    if parts[2] not in ("train", "eval", "extract", "heads"):
        raise SystemExit(f"kind must be train, eval, extract, or heads: {line}")
    if len(parts) > 4 and parts[4] not in ("FAILED", "CANCELLED"):
        raise SystemExit(f"terminal must be FAILED or CANCELLED: {line}")
    rows.append("\t".join(parts))
if not rows:
    raise SystemExit("CELLS file contains no retry rows")
print("\n".join(rows))
PY
)" || exit 2
mapfile -t RETRY_SPECS <<< "$RETRY_SPECS_RAW"

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

append_events() {
    python - "$@" <<'PY'
import json, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from src.experiment_tracking import lifecycle

root = Path(sys.argv[2])
attempt_id, fold, fold_dir = sys.argv[3:6]
events = json.loads(sys.argv[6])
for event in events:
    payload = lifecycle.new_job_event(
        job_key=event["job_key"], job_type=event["job_type"], event_type=event["event_type"],
        attempt_id=attempt_id, fold=int(fold), slurm_job_id=event.get("slurm_job_id"),
        dependency_job_ids=event.get("dependency_job_ids"), status=event.get("status"),
        reason=event.get("reason"), resubmission_of_job_id=event.get("resubmission_of_job_id"),
    )
    lifecycle.append_job_event(Path(fold_dir) / "jobs.jsonl", payload)
PY
}

train_index=0
aux_index=0
declare -a train_lanes aux_lanes
retry_registry="$SUBMISSIONS_ROOT/$RUN_ID/retry_${RETRY_TAG}_jobs.tsv"
if [ "$DRY_RUN" = 0 ]; then
    [ ! -e "$retry_registry" ] || { echo "Refusing existing retry registry: $retry_registry" >&2; exit 4; }
    mkdir -p "$(dirname "$retry_registry")"
    printf 'backbone\tmodality\tkind\tjob_id\tdependency\tattempt_id\n' > "$retry_registry"
fi

for spec in "${RETRY_SPECS[@]}"; do
    IFS=$'\t' read -r backbone modality kind failed_job terminal <<< "$spec"
    terminal="${terminal:-FAILED}"
    context_path="$CONTEXTS_ROOT/$RUN_ID/$backbone/$modality/fold_0/context.json"
    [ -f "$context_path" ] || { echo "Missing original context: $context_path" >&2; exit 3; }
    old_context="$(cat "$context_path")"
    old_attempt_id="$(printf '%s' "$old_context" | python -c 'import json,sys; print(json.load(sys.stdin)["attempt_id"])')"
    fold_0="$(printf '%s' "$old_context" | python -c 'import json,sys; print(json.load(sys.stdin)["fold"])')"

    # Resolve the cell's config, run root, and backend environment from the
    # campaign matrix.
    config="$("$PROJECT_ROOT/scripts/../scripts/.daic_officialdev_resolve_config.sh" "$backbone" "$modality" "$PROJECT_ROOT" 2>/dev/null || true)"
    if [ -z "$config" ]; then
        config="$(python - "$MATRIX" "$backbone" "$modality" "$PROJECT_ROOT" <<'PY'
import sys, yaml
from pathlib import Path
matrix = yaml.safe_load(Path(sys.argv[1]).read_text(encoding="utf-8"))
for item in matrix.get("experiments") or []:
    if item.get("backbone") == sys.argv[2] and item.get("modality") == sys.argv[3]:
        print(str(Path(sys.argv[4]) / item["config"]))
        break
PY
)"
    fi
    [ -n "$config" ] || { echo "Cannot resolve config for $backbone/$modality" >&2; exit 3; }

    cell_info="$(python - "$config" "$backbone" "$PROJECT_ROOT" <<'PY'
import sys, yaml
from pathlib import Path
config_path, backbone, root = sys.argv[1:4]
config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
run_root = str(config["output_dirs"]["run_root"]).replace("${PROJECT_ROOT}", root)
env_name = "gemma4_12b_tf5_14_1" if backbone == "gemma4" else "qwen_mn5_rebuilt"
print("\t".join((run_root, env_name)))
PY
)"
    IFS=$'\t' read -r run_root env_name <<< "$cell_info"
    case "$env_name" in
        qwen_mn5_rebuilt) ENV_ACTIVATE="$QWEN_ENV_ACTIVATE";;
        gemma4_12b_tf5_14_1) ENV_ACTIVATE="$GEMMA_ENV_ACTIVATE";;
        *) echo "Unknown backend environment: $env_name" >&2; exit 3;;
    esac
    case "$backbone:$modality" in
        qwen:text_only) MODEL_PATH="$MODEL_PATH_QWEN_TEXT";;
        qwen:*) MODEL_PATH="$MODEL_PATH_QWEN";;
        gemma4:*) MODEL_PATH="$MODEL_PATH_GEMMA4";;
        *) echo "Unknown backbone/modality: $backbone/$modality" >&2; exit 3;;
    esac

    # The original run dir name is recorded in the campaign submission
    # registry of the original run.
    orig_registry="$SUBMISSIONS_ROOT/$RUN_ID/jobs.tsv"
    [ -f "$orig_registry" ] || { echo "Missing original submission registry: $orig_registry" >&2; exit 3; }
    old_run_name="$(awk -F'\t' -v b="$backbone" -v m="$modality" '$1==b && $2==m && $3=="train" {print $6; exit}' "$orig_registry")"
    [ -n "$old_run_name" ] || { echo "Original run name not found in $orig_registry for $backbone/$modality" >&2; exit 3; }
    old_fold_dir="$run_root/$old_run_name/fold_0"

    new_tag="${RETRY_TAG}"
    new_run_name="daic_officialdev_${backbone}_${modality}_seed1337_${RUN_ID}_${MERGE_SHA:0:8}_${new_tag}"
    new_fold_dir="$run_root/$new_run_name/fold_0"
    child_root="$PROJECT_ROOT/output_model/harmonized_v1_officialdev_heads"
    [ "$backbone" = "gemma4" ] && child_root="$PROJECT_ROOT/output_model/harmonized_v1_gemma4_officialdev_heads"
    child_run_name="daic_officialdev_${backbone}_${modality}_fixed_heads_seed1337_${RUN_ID}_${MERGE_SHA:0:8}_${new_tag}"
    child_attempt_dir="$child_root/$modality/daic/$child_run_name/fold_0"
    new_context_dir="$CONTEXTS_ROOT/$RUN_ID/$backbone/$modality/fold_0/${new_tag}"
    new_context_path="$new_context_dir/context.json"

    if [ "$DRY_RUN" = 0 ]; then
        [ ! -e "$new_context_path" ] || { echo "Refusing existing retry context: $new_context_path" >&2; exit 4; }
        [ ! -e "$new_fold_dir" ] || { echo "Refusing existing retry run dir: $new_fold_dir" >&2; exit 4; }
        mkdir -p "$new_context_dir"
        python - "$new_context_path" "$PREFLIGHT_AUDIT" "$RUN_ID" "$backbone" "$modality" "$new_run_name" "$PROJECT_ROOT" "$GITHUB_ISSUE" "$GITHUB_PR" "$GROUP_ID" "$old_attempt_id" "$fold_0" <<'PY'
import json, sys
from pathlib import Path
sys.path.insert(0, sys.argv[8])
from src.experiment_tracking.identity import new_attempt_id
context_path, audit_path, run_id, backbone, modality, run_name, root, github_issue, github_pr, group_id, supersedes, fold = sys.argv[1:]
audit = json.load(open(audit_path, encoding="utf-8"))
commit = str(audit.get("source_commit") or "")
if len(commit) != 40:
    raise SystemExit("Preflight audit has no full source commit")
logical = f"daic_officialdev_{backbone}_{modality}_seed1337"
payload = {
    "schema_version": "audiollm.experiment_context.v1",
    "group_id": group_id,
    "logical_run_name": logical,
    "attempt_id": new_attempt_id(logical, commit),
    "fold": int(fold),
    "seed": 1337,
    "supersedes_attempt_id": supersedes,
    "source": {"git_commit": commit, "git_branch": audit.get("source_branch"), "git_dirty": False},
    "research": {"github_issue": int(github_issue), "github_pr": int(github_pr)},
    "hashes": {"manifest_sha256": audit["split_audit"]["manifest_sha256"], "split_sha256": audit["split_audit"]["partition_file_sha256"]},
    "slurm": {"train_job_id": None, "eval_job_ids": []},
}
path = Path(context_path)
path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY
        # Append the terminal event for the failed job on the original attempt.
        append_events "$PROJECT_ROOT" "$old_attempt_id" "$fold_0" "$old_fold_dir" "$(python - "$terminal" "$kind" "$failed_job" <<'PY'
import json, sys
terminal, kind, failed_job = sys.argv[1:4]
job_type = {"train": "train", "eval": "evaluation", "extract": "hidden_extraction", "heads": "hidden_classifier"}[kind]
print(json.dumps([{
    "job_key": kind, "job_type": job_type, "event_type": terminal,
    "slurm_job_id": failed_job, "status": terminal,
    "reason": "recorded by retry launcher",
}]))
PY
)"
    fi

    # Submit the affected jobs. For a train failure: all four with the new
    # attempt. Otherwise: from the failed kind onward against the existing
    # training attempt or a new child attempt.
    resubmit_all="0"
    [ "$kind" = "train" ] && resubmit_all="1"

    train_job=""
    if [ "$resubmit_all" = "1" ]; then
        train_lane=$((train_index % MAX_CONCURRENT_TRAINS))
        train_throttle="${train_lanes[$train_lane]:-}"
        train_dep="$(dependency_arg "" "$train_throttle" || true)"
        export_spec="ALL,PROJECT_ROOT=$PROJECT_ROOT,CONFIG=$config,FOLD=$fold_0,RUN_NAME=$new_run_name,SKIP_MANIFEST_BUILD=1,EXPERIMENT_CONTEXT=$new_context_path,ENV_ACTIVATE=$ENV_ACTIVATE"
        train_cmd=(sbatch --parsable --job-name="od-${backbone:0:3}-${modality:0:2}-tr-${RETRY_TAG}" --export="$export_spec")
        [ -n "$train_dep" ] && train_cmd+=("$train_dep")
        train_cmd+=("$TRAIN_WORKER")
        train_raw="$(submit "${train_cmd[@]}")"
        train_job="$(job_id "$train_raw")"
        train_lanes[$train_lane]="$train_job"
        train_index=$((train_index + 1))
        if [ "$DRY_RUN" = 0 ]; then
            printf '%s\t%s\ttrain\t%s\t%s\t%s\n' "$backbone" "$modality" "$train_job" "$train_throttle" "$old_attempt_id" >> "$retry_registry"
        fi
    fi

    chain_job="$train_job"
    eval_job=""
    if [ "$resubmit_all" = "1" ] || [ "$kind" = "eval" ]; then
        aux_lane=$((aux_index % MAX_CONCURRENT_AUX))
        aux_throttle="${aux_lanes[$aux_lane]:-}"
        eval_dep="$(dependency_arg "$chain_job" "$aux_throttle")"
        eval_export="ALL,PROJECT_ROOT=$PROJECT_ROOT,CONFIG=$config,FOLD=$fold_0,SKIP_MANIFEST_BUILD=1,ENV_ACTIVATE=$ENV_ACTIVATE"
        if [ "$resubmit_all" = "1" ]; then
            eval_export="$eval_export,EXPERIMENT_CONTEXT=$new_context_path"
            eval_checkpoint_dir="$new_fold_dir/best_model"
        else
            eval_export="$eval_export,EXPERIMENT_CONTEXT=$context_path"
            eval_checkpoint_dir="$old_fold_dir/best_model"
        fi
        eval_cmd=(sbatch --parsable --job-name="od-${backbone:0:3}-${modality:0:2}-ev-${RETRY_TAG}" "$eval_dep" --export="$eval_export,CHECKPOINT_DIR=$eval_checkpoint_dir,OUTPUT_DIR=$eval_checkpoint_dir/standalone_eval" "$EVAL_WORKER")
        eval_raw="$(submit "${eval_cmd[@]}")"
        eval_job="$(job_id "$eval_raw")"
        chain_job="$eval_job"
        aux_lanes[$aux_lane]="$chain_job"
        aux_index=$((aux_index + 1))
        if [ "$DRY_RUN" = 0 ]; then
            printf '%s\t%s\teval\t%s\t%s\t%s\n' "$backbone" "$modality" "$eval_job" "$train_job" "$old_attempt_id" >> "$retry_registry"
            if [ "$resubmit_all" = "0" ]; then
                append_events "$PROJECT_ROOT" "$old_attempt_id" "$fold_0" "$old_fold_dir" "$(python - "$eval_job" "$train_job" <<'PY'
import json, sys
print(json.dumps([{
    "job_key": "best_eval", "job_type": "evaluation", "event_type": "SUBMITTED",
    "slurm_job_id": sys.argv[1], "dependency_job_ids": [sys.argv[2]] if sys.argv[2] else [],
    "status": "PENDING", "reason": "resubmitted by retry launcher",
    "resubmission_of_job_id": None,
}]))
PY
)"
            fi
        fi
    fi

    parent_dir_for_child="$new_fold_dir"
    [ "$resubmit_all" = "1" ] || parent_dir_for_child="$old_fold_dir"
    child_parent_attempt="$old_attempt_id"
    extract_job=""
    if [ "$resubmit_all" = "1" ] || [ "$kind" = "eval" ] || [ "$kind" = "extract" ]; then
        aux_lane=$((aux_index % MAX_CONCURRENT_AUX))
        aux_throttle="${aux_lanes[$aux_lane]:-}"
        extract_dep="$(dependency_arg "$chain_job" "$aux_throttle")"
        extract_cmd=(sbatch --parsable --job-name="od-${backbone:0:3}-${modality:0:2}-ex-${RETRY_TAG}" "$extract_dep" --export="ALL,PROJECT_ROOT=$PROJECT_ROOT,ENV_ACTIVATE=$ENV_ACTIVATE,ATTEMPT_DIR=$child_attempt_dir,PARENT_FOLD_DIR=$parent_dir_for_child,MODEL_PATH=$MODEL_PATH,MODALITY=$modality,BACKBONE=$backbone,RUN_NAME=$child_run_name,GROUP_ID=$GROUP_ID,MERGED_SHA=$MERGE_SHA,BRANCH=$MERGE_BRANCH,PR_NUMBER=$GITHUB_PR,CONDITION=daic_officialdev" "$EXTRACT_WORKER")
        extract_raw="$(submit "${extract_cmd[@]}")"
        extract_job="$(job_id "$extract_raw")"
        chain_job="$extract_job"
        aux_lanes[$aux_lane]="$chain_job"
        aux_index=$((aux_index + 1))
        if [ "$DRY_RUN" = 0 ]; then
            printf '%s\t%s\textract\t%s\t%s\t%s\n' "$backbone" "$modality" "$extract_job" "$chain_job" "$old_attempt_id" >> "$retry_registry"
        fi
    fi

    heads_job=""
    if [ "$resubmit_all" = "1" ] || [ "$kind" = "eval" ] || [ "$kind" = "extract" ] || [ "$kind" = "heads" ]; then
        aux_lane=$((aux_index % MAX_CONCURRENT_AUX))
        aux_throttle="${aux_lanes[$aux_lane]:-}"
        heads_dep="$(dependency_arg "$chain_job" "$aux_throttle")"
        heads_cmd=(sbatch --parsable --job-name="od-${backbone:0:3}-${modality:0:2}-hd-${RETRY_TAG}" "$heads_dep" --export="ALL,PROJECT_ROOT=$PROJECT_ROOT,ATTEMPT_DIR=$child_attempt_dir,PARENT_FOLD_DIR=$parent_dir_for_child" "$HEADS_WORKER")
        heads_raw="$(submit "${heads_cmd[@]}")"
        heads_job="$(job_id "$heads_raw")"
        aux_lanes[$aux_lane]="$heads_job"
        aux_index=$((aux_index + 1))
        if [ "$DRY_RUN" = 0 ]; then
            printf '%s\t%s\theads\t%s\t%s\t%s\n' "$backbone" "$modality" "$heads_job" "$extract_job" "$old_attempt_id" >> "$retry_registry"
        fi
    fi

    if [ "$DRY_RUN" = 0 ] && [ "$resubmit_all" = "1" ]; then
        python - "$new_context_path" "$new_fold_dir" "$train_job" "$eval_job" "$PROJECT_ROOT" <<'PY'
import json, os, sys
from pathlib import Path
sys.path.insert(0, sys.argv[5])
from src.experiment_tracking import lifecycle

context_path, fold_dir, train_job, eval_job = sys.argv[1:5]
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
        resubmission_of_job_id=None,
    ),
]
if eval_job:
    events.append(lifecycle.new_job_event(
        job_key="best_eval", job_type="evaluation", event_type="SUBMITTED",
        attempt_id=context["attempt_id"], fold=int(context["fold"]),
        slurm_job_id=eval_job, dependency_job_ids=[train_job], status="PENDING",
    ))
for event in events:
    lifecycle.append_job_event(run_root / "jobs.jsonl", event)
PY
    fi
done

echo "Officialdev retry plan: cells=${#RETRY_SPECS[@]} retry_tag=$RETRY_TAG dry_run=$DRY_RUN"
[ "$DRY_RUN" = 1 ] || echo "Retry registry: $retry_registry"
