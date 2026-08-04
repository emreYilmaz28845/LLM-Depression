#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-/gpfs/projects/etur92/ozu647717/AudioLLM/LLM-Depression}"
MATRIX_PATH="${MATRIX_PATH:?Set MATRIX_PATH}"
DRY_RUN="${DRY_RUN:-1}"
RESUME="${RESUME:-0}"
RUN_GROUPS="${RUN_GROUPS:-joint,independent}"
MAX_CONCURRENT_TRAIN="${MAX_CONCURRENT_TRAIN:-4}"
MAX_CONCURRENT_EVAL="${MAX_CONCURRENT_EVAL:-4}"
for value in "$DRY_RUN" "$RESUME"; do case "$value" in 0|1) ;; *) echo "DRY_RUN and RESUME must be 0 or 1" >&2; exit 2;; esac; done
[ -f "$MATRIX_PATH" ] || { echo "Missing matrix: $MATRIX_PATH" >&2; exit 3; }
readarray -t META < <(python - "$MATRIX_PATH" "$RUN_GROUPS" <<'PY'
import json, sys
p=json.load(open(sys.argv[1]))
groups={value.strip() for value in sys.argv[2].split(',') if value.strip()}
for group in ('joint','independent'):
    indices=[str(i) for i,t in enumerate([x for x in p['tasks'] if x['kind']=='train']) if t['group']==group and group in groups]
    print(','.join(indices))
print(p['run_id']); print(p['stage'])
PY
)
submit() {
  if [ "$DRY_RUN" = 1 ]; then
    printf 'DRY_RUN ' >&2; printf '%q ' "$@" >&2; printf '\n' >&2
    printf 'dry_%s\n' "$RANDOM"
  else
    "$@"
  fi
}
job_id() { printf '%s' "${1%%;*}"; }
LOG_ROOT="$PROJECT_ROOT/logs/daic_non_k/${META[2]}/${META[3]}/arrays"
mkdir -p "$LOG_ROOT"
declare -a GROUP_RECORDS=()
for pair in "joint:${META[0]}" "independent:${META[1]}"; do
  group="${pair%%:*}"; indices="${pair#*:}"
  [ -n "$indices" ] || continue
  train_raw="$(submit sbatch --parsable --array="$indices%$MAX_CONCURRENT_TRAIN" --gres=gpu:4 --ntasks=4 --ntasks-per-node=4 --cpus-per-task=20 \
    --export="ALL,PROJECT_ROOT=$PROJECT_ROOT,MATRIX_PATH=$MATRIX_PATH,TASK_KIND=train,ARRAY_LOG_ROOT=$LOG_ROOT,RESUME=$RESUME" \
    "$PROJECT_ROOT/scripts/run_daic_non_k_array_slurm.sh")"
  train_job="$(job_id "$train_raw")"
  eval_raw="$(submit sbatch --parsable --dependency="afterok:$train_job" --array="$indices%$MAX_CONCURRENT_EVAL" --gres=gpu:1 --ntasks=1 --cpus-per-task=20 --time=24:00:00 \
    --export="ALL,PROJECT_ROOT=$PROJECT_ROOT,MATRIX_PATH=$MATRIX_PATH,TASK_KIND=evaluation,ARRAY_LOG_ROOT=$LOG_ROOT,RESUME=$RESUME" \
    "$PROJECT_ROOT/scripts/run_daic_non_k_array_slurm.sh")"
  eval_job="$(job_id "$eval_raw")"
  GROUP_RECORDS+=("$group|$indices|$train_job|$eval_job")
done
SUBMISSION_PATH="$(dirname "$MATRIX_PATH")/submission_${META[3]}.json"
export SUBMISSION_PATH MATRIX_PATH DRY_RUN RESUME GROUP_RECORDS_TEXT="$(printf '%s\n' "${GROUP_RECORDS[@]}")"
python - <<'PY'
import json, os
from pathlib import Path
matrix=json.loads(Path(os.environ['MATRIX_PATH']).read_text())
groups={}
for line in os.environ.get('GROUP_RECORDS_TEXT','').splitlines():
    if not line: continue
    group, indices, train, evaluation=line.split('|')
    groups[group]={'indices':[int(x) for x in indices.split(',')], 'train_job_id':train, 'evaluation_job_id':evaluation}
payload={'schema_version':'daic_non_k_submission.v1','run_id':matrix['run_id'],'stage':matrix['stage'],
         'matrix_path':os.environ['MATRIX_PATH'],'matrix_hash':matrix['matrix_hash'],
         'dry_run':os.environ['DRY_RUN']=='1','resume':os.environ['RESUME']=='1','groups':groups}
Path(os.environ['SUBMISSION_PATH']).write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n')
print(json.dumps(payload,sort_keys=True))
PY
