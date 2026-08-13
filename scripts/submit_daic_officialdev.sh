#!/usr/bin/env bash
# DAIC official-development Qwen + Gemma 4 campaign launcher.
#
# Submits the six-cell chain graph (train -> official-dev eval -> hidden
# extraction -> CPU fixed heads) with backend-correct environments, minted
# experiment contexts, immediate SUBMITTED events, raw sbatch responses in a
# task-owned audit, and the combined project-wide 64-H100 ceiling enforced
# against live allocations under the worst case. Dry-run by default; dry-run
# performs zero mutation.
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/gpfs/projects/etur92/ozu647717/AudioLLM/LLM-Depression}"
MATRIX="${MATRIX:-$PROJECT_ROOT/configs/experiments/daic_officialdev/matrix.yaml}"
RUN_ID="${RUN_ID:?Set a unique campaign RUN_ID}"
DRY_RUN="${DRY_RUN:-1}"
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
if [ "$MAX_CONCURRENT_TRAINS" -lt 1 ] || [ "$MAX_CONCURRENT_AUX" -lt 1 ]; then
    echo "Concurrency limits must be positive." >&2
    exit 2
fi
if [ $((MAX_CONCURRENT_TRAINS * 4 + MAX_CONCURRENT_AUX)) -gt "$GPU_CEILING" ]; then
    echo "Requested lanes can exceed the $GPU_CEILING-GPU project ceiling: trains=$MAX_CONCURRENT_TRAINS aux=$MAX_CONCURRENT_AUX" >&2
    exit 2
fi
for path in "$MATRIX" "$TRAIN_WORKER" "$EVAL_WORKER" "$EXTRACT_WORKER" "$HEADS_WORKER"; do
    [ -f "$path" ] || { echo "Missing required file: $path" >&2; exit 3; }
done

cd "$PROJECT_ROOT"
MERGE_SHA="$(tr -d '\n' < "$PROJECT_ROOT/.provenance/git_commit.txt" 2>/dev/null || git rev-parse HEAD)"
MERGE_BRANCH="$(tr -d '\n' < "$PROJECT_ROOT/.provenance/git_branch.txt" 2>/dev/null || git rev-parse --abbrev-ref HEAD)"
GROUP_ID="daic-officialdev-qwen-gemma-v1-${MERGE_SHA:0:12}"
CAMPAIGN_TAG="${RUN_ID}_${MERGE_SHA:0:8}"

if [ "$DRY_RUN" = 0 ]; then
    python - "$PREFLIGHT_AUDIT" "$RUN_ID" "$MERGE_SHA" <<'PY'
import json, sys
payload = json.load(open(sys.argv[1], encoding="utf-8"))
if payload.get("status") != "passed" or payload.get("run_id") != sys.argv[2]:
    raise SystemExit(f"Incompatible MN5 preflight audit: {sys.argv[1]}")
if payload.get("source_commit") != sys.argv[3]:
    raise SystemExit(f"Preflight audit source commit does not match the deployed source: {sys.argv[3]}")
scope = payload.get("job_scope") or {}
if scope.get("principal_jobs") != 24:
    raise SystemExit(f"Preflight audit must declare exactly 24 principal jobs: {scope}")
if len(payload.get("configs") or []) != 6:
    raise SystemExit(f"Preflight audit must cover the six officialdev configs: {payload.get('configs')}")
PY
fi

# Live worst-case allocation check: current user GPU allocations plus the
# maximum this campaign could add must stay at or below the combined ceiling.
current_alloc=0
while IFS= read -r tres; do
    gpu="$(printf '%s' "$tres" | sed -n 's/.*gres\/gpu=\([0-9][0-9]*\).*/\1/p')"
    if [ -n "$gpu" ]; then
        current_alloc=$((current_alloc + gpu))
    fi
done < <(squeue -h -r -u "$USER" -o "%b" 2>/dev/null || true)
requested=$((MAX_CONCURRENT_TRAINS * 4 + MAX_CONCURRENT_AUX))
worst_case=$((current_alloc + requested))
echo "GPU allocation check: currently_allocated=$current_alloc requested_by_campaign=$requested worst_case=$worst_case ceiling=$GPU_CEILING"
if [ "$worst_case" -gt "$GPU_CEILING" ]; then
    echo "Worst-case combined allocation exceeds the $GPU_CEILING-H100 project ceiling." >&2
    exit 2
fi

mapfile -t TASKS < <(python - "$MATRIX" "$PROJECT_ROOT" <<'PY'
import sys, yaml
from pathlib import Path
root = Path(sys.argv[2])
matrix = yaml.safe_load(Path(sys.argv[1]).read_text(encoding="utf-8"))
if matrix.get("seed") != 1337 or matrix.get("folds") != [0]:
    raise SystemExit("Officialdev matrix must use seed 1337 and fold 0 only")
if matrix.get("fixed_heads") != ["logreg_raw", "xgb_raw"]:
    raise SystemExit("Officialdev matrix must contain only logreg_raw and xgb_raw heads")
if matrix.get("checkpoint_selection") != "inner_val_macro_f1":
    raise SystemExit("Officialdev matrix must select by inner_val_macro_f1")
if matrix.get("separate_final_eval") is not True or matrix.get("run_final_eval_in_train") is not False:
    raise SystemExit("Officialdev matrix must use a separate final evaluation")
if matrix.get("principal_jobs") != 24:
    raise SystemExit(f"Officialdev matrix must declare 24 principal jobs, got {matrix.get('principal_jobs')}")
experiments = matrix.get("experiments") or []
if len(experiments) != 6:
    raise SystemExit(f"Officialdev matrix must contain exactly six cells, got {len(experiments)}")
for item in experiments:
    config_path = root / item["config"]
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("dataset") != "daic":
        raise SystemExit(f"Officialdev cell must be daic: {config_path}")
    if int(config["training"]["num_train_epochs"]) != 20:
        raise SystemExit(f"Expected 20 epochs: {config_path}")
    if config["training"]["selection_metric"] != "inner_val_macro_f1":
        raise SystemExit(f"Expected macro-F1 selection: {config_path}")
    if config["training"].get("run_final_eval_in_train") is not False:
        raise SystemExit(f"Officialdev training must not evaluate final dev: {config_path}")
    if config["evaluation"].get("evaluation_view") != "harmonized_all_windows_full_coverage":
        raise SystemExit(f"Officialdev eval view mismatch: {config_path}")
    if config["split"].get("final_eval_partition") != "val" or "selection_partition" in config["split"]:
        raise SystemExit(f"Officialdev split contract mismatch: {config_path}")
    modality = "audio_text" if config["data"].get("use_audio") and config["data"].get("use_text") else "audio_only" if config["data"].get("use_audio") else "text_only"
    run_root = str(config["output_dirs"]["run_root"]).replace("${PROJECT_ROOT}", str(root))
    print("\t".join((str(config_path), item["backbone"], item["env"], modality, "0", run_root)))
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
raw_log="$SUBMISSIONS_ROOT/$RUN_ID/jobs_raw.jsonl"
if [ "$DRY_RUN" = 0 ]; then
    [ ! -e "$registry" ] || { echo "Refusing existing submission registry: $registry" >&2; exit 4; }
    mkdir -p "$(dirname "$registry")"
    printf 'backbone\tmodality\tkind\tjob_id\tdependency\trun_name\n' > "$registry"
    : > "$raw_log"
fi

echo "Campaign: group_id=$GROUP_ID merge_sha=$MERGE_SHA branch=$MERGE_BRANCH issue=$GITHUB_ISSUE pr=$GITHUB_PR"
for task in "${TASKS[@]}"; do
    IFS=$'\t' read -r config backbone env_name modality fold run_root <<< "$task"
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
    run_name="daic_officialdev_${backbone}_${modality}_seed1337_${CAMPAIGN_TAG}"
    fold_dir="$run_root/$run_name/fold_$fold"
    case "$backbone" in
        qwen) child_root="$PROJECT_ROOT/output_model/harmonized_v1_officialdev_heads";;
        gemma4) child_root="$PROJECT_ROOT/output_model/harmonized_v1_gemma4_officialdev_heads";;
        *) echo "Unknown backbone: $backbone" >&2; exit 3;;
    esac
    child_run_name="daic_officialdev_${backbone}_${modality}_fixed_heads_seed1337_${CAMPAIGN_TAG}"
    child_attempt_dir="$child_root/$modality/daic/$child_run_name/fold_0"
    context_dir="$CONTEXTS_ROOT/$RUN_ID/$backbone/$modality/fold_$fold"
    context_path="$context_dir/context.json"
    if [ "$DRY_RUN" = 0 ]; then
        for existing in "$context_path" "$fold_dir" "$child_attempt_dir"; do
            if [ -e "$existing" ]; then
                echo "Refusing existing destination: $existing" >&2
                exit 4
            fi
        done
        mkdir -p "$context_dir"
        python - "$context_path" "$PREFLIGHT_AUDIT" "$RUN_ID" "$backbone" "$modality" "$fold" "$run_name" "$PROJECT_ROOT" "$GITHUB_ISSUE" "$GITHUB_PR" "$GROUP_ID" <<'PY'
import json, sys
from pathlib import Path
sys.path.insert(0, sys.argv[8])
from src.experiment_tracking.identity import new_attempt_id
context_path, audit_path, run_id, backbone, modality, fold, run_name, root, github_issue, github_pr, group_id = sys.argv[1:]
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
    "source": {"git_commit": commit, "git_branch": audit.get("source_branch"), "git_dirty": False},
    "research": {"github_issue": int(github_issue), "github_pr": int(github_pr)},
    "hashes": {"manifest_sha256": audit["split_audit"]["manifest_sha256"], "split_sha256": audit["split_audit"]["partition_file_sha256"]},
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
    export_spec="ALL,PROJECT_ROOT=$PROJECT_ROOT,CONFIG=$config,FOLD=$fold,RUN_NAME=$run_name,SKIP_MANIFEST_BUILD=1,EXPERIMENT_CONTEXT=$context_path,ENV_ACTIVATE=$ENV_ACTIVATE"
    train_cmd=(sbatch --parsable --job-name="od-${backbone:0:3}-${modality:0:2}-tr" --export="$export_spec")
    [ -n "$train_dep" ] && train_cmd+=("$train_dep")
    train_cmd+=("$TRAIN_WORKER")
    train_raw="$(submit "${train_cmd[@]}")"
    train_job="$(job_id "$train_raw")"
    train_lanes[$train_lane]="$train_job"
    train_index=$((train_index + 1))
    if [ "$DRY_RUN" = 0 ]; then
        printf '%s\t%s\ttrain\t%s\t%s\t%s\n' "$backbone" "$modality" "$train_job" "$train_throttle" "$run_name" >> "$registry"
        printf '%s\n' "$(printf '%s' "$train_raw" | jq -R . 2>/dev/null || printf '"%s"' "$train_raw")" >> "$raw_log"
    fi

    chain_job="$train_job"
    aux_lane=$((aux_index % MAX_CONCURRENT_AUX))
    aux_throttle="${aux_lanes[$aux_lane]:-}"
    eval_dep="$(dependency_arg "$train_job" "$aux_throttle")"
    eval_cmd=(sbatch --parsable --job-name="od-${backbone:0:3}-${modality:0:2}-ev" "$eval_dep" --export="$export_spec,CHECKPOINT_DIR=$fold_dir/best_model,OUTPUT_DIR=$fold_dir/best_model/standalone_eval" "$EVAL_WORKER")
    eval_raw="$(submit "${eval_cmd[@]}")"
    eval_job="$(job_id "$eval_raw")"
    chain_job="$eval_job"
    aux_lanes[$aux_lane]="$chain_job"
    aux_index=$((aux_index + 1))
    if [ "$DRY_RUN" = 0 ]; then
        printf '%s\t%s\teval\t%s\t%s,%s\t%s\n' "$backbone" "$modality" "$eval_job" "$train_job" "$aux_throttle" "$run_name" >> "$registry"
        printf '%s\n' "$(printf '%s' "$eval_raw" | jq -R . 2>/dev/null || printf '"%s"' "$eval_raw")" >> "$raw_log"
    fi

    aux_lane=$((aux_index % MAX_CONCURRENT_AUX))
    aux_throttle="${aux_lanes[$aux_lane]:-}"
    extract_dep="$(dependency_arg "$chain_job" "$aux_throttle")"
    extract_cmd=(sbatch --parsable --job-name="od-${backbone:0:3}-${modality:0:2}-ex" "$extract_dep" --export="ALL,PROJECT_ROOT=$PROJECT_ROOT,ENV_ACTIVATE=$ENV_ACTIVATE,ATTEMPT_DIR=$child_attempt_dir,PARENT_FOLD_DIR=$fold_dir,MODEL_PATH=$MODEL_PATH,MODALITY=$modality,BACKBONE=$backbone,RUN_NAME=$child_run_name,GROUP_ID=$GROUP_ID,MERGED_SHA=$MERGE_SHA,BRANCH=$MERGE_BRANCH,PR_NUMBER=$GITHUB_PR,CONDITION=daic_officialdev" "$EXTRACT_WORKER")
    extract_raw="$(submit "${extract_cmd[@]}")"
    extract_job="$(job_id "$extract_raw")"
    aux_lanes[$aux_lane]="$extract_job"
    aux_index=$((aux_index + 1))
    if [ "$DRY_RUN" = 0 ]; then
        printf '%s\t%s\textract\t%s\t%s,%s\t%s\n' "$backbone" "$modality" "$extract_job" "$chain_job" "$aux_throttle" "$child_run_name" >> "$registry"
        printf '%s\n' "$(printf '%s' "$extract_raw" | jq -R . 2>/dev/null || printf '"%s"' "$extract_raw")" >> "$raw_log"
    fi

    aux_lane=$((aux_index % MAX_CONCURRENT_AUX))
    aux_throttle="${aux_lanes[$aux_lane]:-}"
    heads_dep="$(dependency_arg "$extract_job" "$aux_throttle")"
    heads_cmd=(sbatch --parsable --job-name="od-${backbone:0:3}-${modality:0:2}-hd" "$heads_dep" --export="ALL,PROJECT_ROOT=$PROJECT_ROOT,ATTEMPT_DIR=$child_attempt_dir,PARENT_FOLD_DIR=$fold_dir" "$HEADS_WORKER")
    heads_raw="$(submit "${heads_cmd[@]}")"
    heads_job="$(job_id "$heads_raw")"
    aux_lanes[$aux_lane]="$heads_job"
    aux_index=$((aux_index + 1))
    if [ "$DRY_RUN" = 0 ]; then
        printf '%s\t%s\theads\t%s\t%s,%s\t%s\n' "$backbone" "$modality" "$heads_job" "$extract_job" "$aux_throttle" "$child_run_name" >> "$registry"
        printf '%s\n' "$(printf '%s' "$heads_raw" | jq -R . 2>/dev/null || printf '"%s"' "$heads_raw")" >> "$raw_log"
        python - "$context_path" "$fold_dir" "$train_job" "$eval_job" "$PROJECT_ROOT" <<'PY'
import json, os, sys
from pathlib import Path
sys.path.insert(0, sys.argv[5])
from src.experiment_tracking import lifecycle

context_path, fold_dir, train_job, eval_job = sys.argv[1:5]
path = Path(context_path)
context = json.loads(path.read_text(encoding="utf-8"))
context["slurm"] = {"train_job_id": train_job, "eval_job_ids": [eval_job]}
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
    ),
    lifecycle.new_job_event(
        job_key="best_eval", job_type="evaluation", event_type="SUBMITTED",
        attempt_id=context["attempt_id"], fold=int(context["fold"]),
        slurm_job_id=eval_job, dependency_job_ids=[train_job], status="PENDING",
    ),
]
for event in events:
    lifecycle.append_job_event(run_root / "jobs.jsonl", event)
PY
    fi
done

echo "Officialdev campaign plan: cells=${#TASKS[@]} train_lanes=$MAX_CONCURRENT_TRAINS aux_lanes=$MAX_CONCURRENT_AUX max_gpus=$requested worst_case_gpus=$worst_case github_issue=$GITHUB_ISSUE github_pr=$GITHUB_PR dry_run=$DRY_RUN"
[ "$DRY_RUN" = 1 ] || echo "Submission registry: $registry"
[ "$DRY_RUN" = 1 ] || echo "Raw submission audit: $raw_log"
