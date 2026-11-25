#!/usr/bin/env bash
# Run a resumable job as a chain of short acc_debug jobs (2 h limit, 1 submitted job per user on MN5).
# Runs on the local machine (bash only, no model tokens): submit one link, wait until it leaves the
# queue, count finished answers on MN5, submit the next link, until the target count is reached.
# The job script must resume from its OUT_DIR (run_qwen_manyshot_explain.py does) and should get
# "--time-budget-min" in EXTRA_ARGS so each link stops cleanly before the time limit.
#
# Exit codes (each one is a point where the agent decides what to do next):
#   0 all answers present        1 a link made no progress (failed / crashed / too slow)
#   2 quota gate or GPU cap blocked a submission                 3 ssh unreachable
#   11 a link stayed PENDING longer than --max-pending hours     12 --max-links reached
#
# Usage: scripts/mn5_debug_chain.sh --name NAME --out-dir REMOTE_OUT --total N --export "VAR=..,VAR=.."
#        [--script scripts/run_qwen_manyshot_explain_slurm.sh] [--workdir REMOTE_CODE] [--logs REMOTE_LOGDIR]
# --export values may contain spaces (quoted on the remote side) but not commas or single quotes.
#        [--time 02:00:00] [--gpus 2] [--max-links 20] [--max-pending 4] [--interval 300] [--host alogin2.bsc.es]
set -uo pipefail
NAME=""; OUT=""; TOTAL=""; EXPORT=""; SCRIPT=scripts/run_qwen_manyshot_explain_slurm.sh; WORKDIR=""; LOGS=""
TIME=02:00:00; GPUS=2; MAX_LINKS=20; MAX_PEND=4; INTERVAL=300; HOST=alogin2.bsc.es
while [ $# -gt 0 ]; do case "$1" in
  --name) NAME="$2";; --out-dir) OUT="$2";; --total) TOTAL="$2";; --export) EXPORT="$2";;
  --script) SCRIPT="$2";; --workdir) WORKDIR="$2";; --logs) LOGS="$2";; --time) TIME="$2";;
  --gpus) GPUS="$2";; --max-links) MAX_LINKS="$2";; --max-pending) MAX_PEND="$2";;
  --interval) INTERVAL="$2";; --host) HOST="$2";;
  -h|--help) sed -n 2,16p "$0"; exit 0;; *) echo "unknown arg $1" >&2; exit 64;; esac; shift 2; done
for v in NAME OUT TOTAL EXPORT WORKDIR LOGS; do [ -n "${!v}" ] || { echo "missing --${v,,}" >&2; exit 64; }; done
REPO=$(cd "$(dirname "$0")/.." && pwd)
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=20 -o ServerAliveInterval=30 "$HOST")
# quota core-hours for one link: GPUs x hours x 20
HOURS=$(awk -F: '{print $1 + $2/60}' <<<"$TIME"); NEED=$(awk -v g="$GPUS" -v h="$HOURS" 'BEGIN{print int(g*h*20+0.5)}')
log() { echo "[chain $NAME $(date -u +%H:%MZ)] $*"; }

remote() {  # retry ssh a few times; exit 3 if MN5 stays unreachable
  local i out
  for i in 1 2 3 4 5 6; do out=$("${SSH[@]}" "$1" 2>/dev/null) && { echo "$out"; return 0; }; sleep 60; done
  log "ssh unreachable"; exit 3
}
count_done() { remote "cat $OUT/responses*.jsonl 2>/dev/null | grep -c subject_id || true" | tail -1; }
gpus_in_use() {  # own running + pending-not-held GPUs, kaan_* excluded (AGENTS.md GPU cap)
  remote "squeue -h -u \$USER -o '%j|%T|%r|%b'" | awk -F'|' '
    $1 ~ /^kaan_/ {next} $3 ~ /Held/ {next}
    {n=0; if (match($4, /gpu[^0-9]*([0-9]+)/, m)) n=m[1]; s+=n} END{print s+0}'
}

done_n=$(count_done); log "start: $done_n/$TOTAL answers present, link cost $NEED quota core-hours"
links=0
while [ "$done_n" -lt "$TOTAL" ]; do
  [ "$links" -ge "$MAX_LINKS" ] && { log "max links ($MAX_LINKS) reached at $done_n/$TOTAL"; exit 12; }
  if ! python "$REPO/.agents/skills/mn5-cluster-ops/scripts/mn5_quota_check.py" --need "$NEED" >/dev/null 2>&1; then
    log "quota gate did not return OK; not submitting"; exit 2; fi
  used=$(gpus_in_use)
  if [ $((used + GPUS)) -gt 8 ]; then log "GPU cap: $used in use + $GPUS > 8; not submitting"; exit 2; fi
  jid=$(remote "cd $WORKDIR && sbatch --parsable -q acc_debug -t $TIME -J $NAME -o $LOGS/${NAME}_%j.out \
        -e $LOGS/${NAME}_%j.err --export='ALL,$EXPORT' $SCRIPT 2>&1 || true" | tail -1)
  if ! [[ "$jid" =~ ^[0-9]+$ ]]; then  # e.g. acc_debug already holds this user's one allowed job
    log "sbatch refused: $jid; retrying in ${INTERVAL}s"; sleep "$INTERVAL"
    blocked=$(( ${blocked:-0} + INTERVAL ))
    [ "$blocked" -gt $(( MAX_PEND * 3600 )) ] && { log "submission blocked > ${MAX_PEND}h"; exit 11; }
    continue
  fi
  blocked=0; links=$((links + 1)); log "link $links submitted: job $jid"
  submitted=$(date +%s)
  while :; do
    st=$(remote "squeue -h -j $jid -o '%T' 2>/dev/null || true" | tail -1)
    [ -z "$st" ] && break
    if [ "$st" = PENDING ] && [ $(( $(date +%s) - submitted )) -gt $(( MAX_PEND * 3600 )) ]; then
      log "job $jid PENDING > ${MAX_PEND}h (left queued)"; exit 11; fi
    sleep "$INTERVAL"
  done
  state=$(remote "sacct -X -n -P -j $jid -o State,Elapsed" | head -1)
  new=$(count_done); log "link $links job $jid ended: $state; answers $done_n -> $new / $TOTAL"
  if [ "$new" -le "$done_n" ]; then log "no progress in job $jid; stopping for a decision"; exit 1; fi
  done_n=$new
done
log "complete: $done_n/$TOTAL answers"; exit 0
