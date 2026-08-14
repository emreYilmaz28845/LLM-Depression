#!/usr/bin/env bash
# Submit the resolved Optuna-100 CPU study matrix (dry-run by default).
#
# Reads the resolved manifest produced by tools/resolve_optuna100_manifest.py.
# For every study: creates the post-hoc attempt (refusing existing
# destinations), marks it deployed, submits one CPU-only study job under a
# concurrency throttle (default at most 20 concurrent studies), and records
# SUBMITTED job events with the raw Slurm job IDs. Optuna consumes no H100s.
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/gpfs/projects/etur92/ozu647717/AudioLLM/LLM-Depression}"
MANIFEST="${MANIFEST:?Set MANIFEST to the resolved manifest JSON}"
RUN_ID="${RUN_ID:?Set a unique RUN_ID}"
DRY_RUN="${DRY_RUN:-1}"
MAX_CONCURRENT_STUDIES="${MAX_CONCURRENT_STUDIES:-20}"
STUDY_WORKER="${STUDY_WORKER:-$PROJECT_ROOT/scripts/run_optuna100_slurm.sh}"
GITHUB_ISSUE="${GITHUB_ISSUE:-}"
GITHUB_PR="${GITHUB_PR:-}"
SUBMISSIONS_ROOT="${SUBMISSIONS_ROOT:-$PROJECT_ROOT/outputs/optuna100_submissions}"

case "$DRY_RUN" in 0|1) ;; *) echo "DRY_RUN must be 0 or 1" >&2; exit 2;; esac
if [ "$MAX_CONCURRENT_STUDIES" -lt 1 ] || [ "$MAX_CONCURRENT_STUDIES" -gt 20 ]; then
    echo "MAX_CONCURRENT_STUDIES must be in [1, 20]." >&2
    exit 2
fi
[ -f "$MANIFEST" ] || { echo "Missing manifest: $MANIFEST" >&2; exit 3; }
[ -f "$STUDY_WORKER" ] || { echo "Missing study worker: $STUDY_WORKER" >&2; exit 3; }

cd "$PROJECT_ROOT"
mapfile -t STUDIES < <(python - "$MANIFEST" <<'PY'
import json, sys
manifest = json.load(open(sys.argv[1], encoding="utf-8"))
studies = manifest.get("studies") or []
missing = [s for s in studies if s.get("cache_missing")]
if missing:
    raise SystemExit(
        f"manifest contains {len(missing)} studies without qualified caches; "
        "refusing to submit. Resolve with --require-caches first."
    )
for study in studies:
    print("\t".join((
        study["backend"], study["dataset"], study["modality"], str(study["fold"]),
        study["attempt_dir"], study["cache_dir"], study["run_name"],
        study.get("group_id", ""),
    )))
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

declare -a study_lanes
study_index=0
registry="$SUBMISSIONS_ROOT/$RUN_ID/jobs.tsv"
if [ "$DRY_RUN" = 0 ]; then
    [ ! -e "$registry" ] || { echo "Refusing existing submission registry: $registry" >&2; exit 4; }
    mkdir -p "$(dirname "$registry")"
    printf 'backend\tdataset\tmodality\tfold\tkind\tjob_id\tdependency\n' > "$registry"
fi

create_count=0
submit_count=0
for study in "${STUDIES[@]}"; do
    IFS=$'\t' read -r backend dataset modality fold attempt_dir cache_dir run_name group_id <<< "$study"
    spec_json="$SUBMISSIONS_ROOT/$RUN_ID/specs/${backend}_${dataset}_${modality}_fold${fold}.json"
    if [ "$DRY_RUN" = 0 ]; then
        mkdir -p "$(dirname "$spec_json")"
        python - "$MANIFEST" "$backend" "$dataset" "$modality" "$fold" "$spec_json" <<'PY'
import json, sys
manifest = json.load(open(sys.argv[1], encoding="utf-8"))
backend, dataset, modality, fold, target = sys.argv[2:7]
for study in manifest["studies"]:
    if (study["backend"], study["dataset"], study["modality"], str(study["fold"])) == (backend, dataset, modality, fold):
        json.dump(study, open(target, "w", encoding="utf-8"), indent=2)
        break
else:
    raise SystemExit("study not found in manifest")
PY
        if [ -e "$attempt_dir" ]; then
            echo "Refusing existing attempt dir: $attempt_dir" >&2
            exit 4
        fi
        python tools/posthoc_head_campaign.py create-attempt \
            --attempt-dir "$attempt_dir" --task-spec "$spec_json" >/dev/null
        python tools/posthoc_head_campaign.py mark-deployed \
            --attempt-dir "$attempt_dir" --reason "attempt deployed with submission wave" >/dev/null
    fi
    create_count=$((create_count + 1))

    lane=$((study_index % MAX_CONCURRENT_STUDIES))
    throttle="${study_lanes[$lane]:-}"
    dep=""
    if [ -n "$throttle" ]; then
        dep="--dependency=afterany:$throttle"
    fi
    export_spec="ALL,PROJECT_ROOT=$PROJECT_ROOT,ATTEMPT_DIR=$attempt_dir,CACHE_DIR=$cache_dir,EXPERIMENT_ID=xgb_optuna100_harmonized_v1,OBJECTIVE=macro_f1,TARGET_TRIALS=100,XGB_THREADS=20"
    if [ "$(python - "$MANIFEST" "$backend" "$dataset" "$modality" "$fold" <<'PY'
import json, sys
manifest = json.load(open(sys.argv[1], encoding="utf-8"))
for study in manifest["studies"]:
    if (study["backend"], study["dataset"], study["modality"], str(study["fold"])) == (sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5]):
        print("1" if study.get("family") == "merged" else "0")
        break
else:
    print("0")
PY
)" = "1" ]; then
        merged_config="$(python - "$MANIFEST" "$backend" "$modality" "$fold" <<'PY'
import json, sys
manifest = json.load(open(sys.argv[1], encoding="utf-8"))
for study in manifest["studies"]:
    if (study["backend"], study["modality"], str(study["fold"])) == (sys.argv[2], sys.argv[3], sys.argv[4]):
        print(study["merged_config"])
        break
PY
)"
        stage="$(python - "$MANIFEST" "$backend" "$modality" "$fold" <<'PY'
import json, sys
manifest = json.load(open(sys.argv[1], encoding="utf-8"))
for study in manifest["studies"]:
    if (study["backend"], study["modality"], str(study["fold"])) == (sys.argv[2], sys.argv[3], sys.argv[4]):
        print(study["stage"])
        break
PY
)"
        export_spec="$export_spec,MERGED=1,MERGED_CONFIG=$merged_config,STAGE=$stage,FOLD=$fold,RUN_ID=$RUN_ID"
    fi
    study_cmd=(sbatch --parsable --job-name="o100-${dataset:0:4}-${modality:0:2}-f$fold")
    [ -n "$dep" ] && study_cmd+=("$dep")
    study_cmd+=(--export="$export_spec" "$STUDY_WORKER")
    study_raw="$(submit "${study_cmd[@]}")"
    study_job="$(job_id "$study_raw")"
    study_lanes[$lane]="$study_job"
    study_index=$((study_index + 1))
    submit_count=$((submit_count + 1))
    if [ "$DRY_RUN" = 0 ]; then
        printf '%s\t%s\t%s\t%s\toptuna\t%s\t%s\n' "$backend" "$dataset" "$modality" "$fold" "$study_job" "$throttle" >> "$registry"
        python tools/posthoc_head_campaign.py record-job \
            --attempt-dir "$attempt_dir" \
            --job-key optuna --job-type hidden_classifier --event-type SUBMITTED \
            --slurm-job-id "$study_job" --job-status PENDING \
            --reason "optuna study submitted" >/dev/null
    fi
done

echo "Optuna-100 plan: studies=${#STUDIES[@]} created=$create_count submitted=$submit_count lanes=$MAX_CONCURRENT_STUDIES gpus=0 dry_run=$DRY_RUN"
[ "$DRY_RUN" = 1 ] || echo "Submission registry: $registry"
