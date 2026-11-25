"""D3TEC zero-shot PHQ-9 item scoring with Claude Opus: the D3TEC port of DAIC zero-shot v4.

Same idea as DAIC v4: the model estimates the participant's answer to each questionnaire item about the
last two weeks, and code computes the label (sum >= 10). D3TEC differs from DAIC in three ways:
  * the gold label is PHQ-9 >= 10 (not PHQ-8), so the model scores all nine PHQ-9 items, item 9 included;
  * the corpus has no official test split, and zero-shot has no training, so every one of the 62
    participants is scored; any calibration is fitted leave-one-participant-out (calibrate_d3tec_phq9.py);
  * the speech is Spanish (Qwen3-ASR transcripts). Each participant answered 27 recorded prompts whose
    wording is not in our metadata, so the text is the 27 responses in prompt order, labelled Response 1-27.
    A few prompts (Responses 16-18) look like read-aloud or fixed-phrase tasks; they are kept, and the prompt says
    some responses may be read text.

Only transcripts leave the machine: no demographics, no diagnosis fields, no IDs other than the target block.
Usage:
  python experiments/manyshot_claude/run_claude_d3tec_phq9.py --out-dir OUT --backend dry-run
  python experiments/manyshot_claude/run_claude_d3tec_phq9.py --out-dir OUT --backend claude-cli \
      --i-confirm-data-transfer-approved
Resumable: subjects already in ``responses.jsonl`` are skipped. ``--metrics-only`` recomputes metrics.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_claude_manyshot import LABEL_TEXT, call_claude_cli, metrics, sha256_file, sha256_text  # noqa: E402

# Harmonized D3TEC manifest (the one the native frozen Qwen runs used); rows are 30 s segments, the text is
# the per-response full transcript, identical across a response's segments (checked).
DEFAULT_MANIFEST = Path("/home/emre/Projects/AudioLLM/LLM-Depression/outputs/manifests_harmonized/d3tec/d3tec_manifest.jsonl")
REFERENCE_MANIFEST_SHA256 = "47a6dfedb1e489a4857d754b599db192b408a3ef5f7beed876e31c5327990da8"
N_RESPONSES = 27

PHQ9_ITEMS = (
    "Little interest or pleasure in doing things",
    "Feeling down, depressed, or hopeless",
    "Trouble falling or staying asleep, or sleeping too much",
    "Feeling tired or having little energy",
    "Poor appetite or overeating",
    "Feeling bad about yourself, or that you are a failure or have let yourself or your family down",
    "Trouble concentrating on things, such as reading the newspaper or watching television",
    "Moving or speaking so slowly that other people could have noticed, or the opposite: being so fidgety or "
    "restless that you have been moving around a lot more than usual",
    "Thoughts that you would be better off dead, or of hurting yourself in some way",
)

# Wording kept as close to DAIC zero-shot v4 as the dataset allows.
SYSTEM_PROMPT = (
    "You are assisting a research study on depression screening from speech transcripts. "
    "Each transcript contains a participant's spoken answers to a series of recorded prompts (D3TEC corpus). "
    "The speech is in Spanish and was transcribed automatically, so it may contain recognition errors. The "
    "prompts themselves are not shown; the answers are numbered Response 1 to Response 27 in the order they were "
    "recorded, and some responses may be read-aloud text rather than free speech. Labels come from the PHQ-9 "
    "questionnaire: Depressed means PHQ-9 >= 10.\n\n"
    "Estimate how the participant would have answered each PHQ-9 item about the last two weeks, using what the "
    "transcript says. Score each item 0 = not at all, 1 = several days, 2 = more than half the days, 3 = nearly "
    "every day. If the transcript has no direct evidence for an item, give your best estimate and say so in the "
    "explanation.\n\nItems:\n" + "\n".join(f"{i}. {t}" for i, t in enumerate(PHQ9_ITEMS, 1)) + "\n\n"
    "Answer with exactly one JSON object and nothing else:\n"
    '{"items": [<item 1 score>, ..., <item 9 score>], "explanation": "<3-6 sentences, in English>"}\n'
    "The explanation must cite the transcript evidence behind the higher item scores."
)


def load_subjects(manifest: Path) -> dict[str, dict]:
    responses: dict[str, dict[int, str]] = defaultdict(dict)
    meta: dict[str, dict] = {}
    for line in manifest.read_text().splitlines():
        row = json.loads(line)
        sid, pid = str(row["subject_id"]), int(row["prompt_id"])
        if row.get("language") != "es":
            raise ValueError(f"{sid}: non-Spanish row.")
        text = row["full_response_transcript"].strip()
        prev = responses[sid].setdefault(pid, text)
        if prev != text:
            raise ValueError(f"{sid} prompt {pid}: segments disagree on the response transcript.")
        rec = {"label": int(row["label"]), "phq9_score": int(row["phq9_score"])}
        if meta.setdefault(sid, rec) != rec:
            raise ValueError(f"{sid}: rows disagree on label/score.")
        if rec["label"] != int(rec["phq9_score"] >= 10):
            raise ValueError(f"{sid}: label is not PHQ-9 >= 10.")
    subjects = {}
    for sid, by_prompt in responses.items():
        if sorted(by_prompt) != list(range(N_RESPONSES)):
            raise ValueError(f"{sid}: expected prompts 0-{N_RESPONSES - 1}, got {sorted(by_prompt)}.")
        text = "\n\n".join(f"Response {p + 1}: {by_prompt[p]}" for p in range(N_RESPONSES))
        subjects[sid] = {"subject_id": sid, **meta[sid], "text": text}
    return subjects


def target_block(target: dict) -> str:
    return "### Participant (score this one)\nTranscript:\n" + target["text"] + "\n\nReturn the JSON object now."


def parse_items_answer(text: str) -> tuple[int | None, str, list[int] | None]:
    for m in reversed(list(re.finditer(r"\{.*?\}", text, flags=re.S))):
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
        items = obj.get("items")
        if (isinstance(items, list) and len(items) == len(PHQ9_ITEMS)
                and all(isinstance(x, int) and not isinstance(x, bool) and 0 <= x <= 3 for x in items)):
            return int(sum(items) >= 10), str(obj.get("explanation", "")).strip(), items
    return None, "", None


def score_metrics(records: list[dict]) -> dict:
    from scipy.stats import pearsonr, spearmanr  # noqa: PLC0415
    from sklearn.metrics import roc_auc_score  # noqa: PLC0415

    ok = [r for r in records if r["phq9_total"] is not None]
    gold = [r["phq9_score"] for r in ok]
    pred = [r["phq9_total"] for r in ok]
    return {"n_scored": len(ok),
            "spearman": round(float(spearmanr(pred, gold)[0]), 4),
            "pearson": round(float(pearsonr(pred, gold)[0]), 4),
            "mae_raw": round(sum(abs(a - b) for a, b in zip(pred, gold)) / len(ok), 3),
            "auroc_total": round(float(roc_auc_score([r["label"] for r in ok], pred)), 4)}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    p.add_argument("--backend", choices=("dry-run", "claude-cli"), default="dry-run")
    p.add_argument("--model", default="claude-opus-5-5")
    p.add_argument("--effort", choices=("low", "medium", "high", "xhigh", "max"), default="medium")
    p.add_argument("--limit", type=int, help="Smoke: only the first N participants.")
    p.add_argument("--timeout", type=int, default=900)
    p.add_argument("--i-confirm-data-transfer-approved", action="store_true",
                   help="Required for claude-cli: D3TEC transcripts leave this machine (user approved 2026-11-25).")
    p.add_argument("--metrics-only", action="store_true")
    args = p.parse_args()

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    resp_path = out / "responses.jsonl"
    digest = sha256_file(args.manifest)
    if digest != REFERENCE_MANIFEST_SHA256:
        sys.exit(f"{args.manifest}: sha256 {digest} is not the reference D3TEC manifest {REFERENCE_MANIFEST_SHA256}.")
    subjects = load_subjects(args.manifest)
    targets = sorted(subjects.values(), key=lambda s: int(s["subject_id"]))
    if args.limit:
        targets = targets[:args.limit]

    run_info = {
        "experiment": "Claude Opus zero-shot D3TEC PHQ-9 item scoring (DAIC v4 port)", "backend": args.backend,
        "model": args.model, "effort": args.effort, "manifest": str(args.manifest), "manifest_sha256": digest,
        "prompt_version": "d3tec_zeroshot_phq9_v1", "system_prompt_sha256": sha256_text(SYSTEM_PROMPT),
        "n_subjects": len(subjects), "n_targets": len(targets),
        "label_counts": {LABEL_TEXT[k]: sum(s["label"] == k for s in subjects.values()) for k in (1, 0)},
        "max_target_chars": max(len(target_block(t)) for t in targets),
        "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    print(json.dumps(run_info, indent=1), flush=True)

    if args.backend == "dry-run" and not args.metrics_only:
        (out / "system_prompt.txt").write_text(SYSTEM_PROMPT)
        (out / "dry_run.json").write_text(json.dumps(run_info, indent=1))
        print(f"[dry-run] wrote the system prompt to {out}; nothing was sent.")
        return

    done = {}
    if resp_path.exists():
        for line in resp_path.read_text().splitlines():
            r = json.loads(line)
            done[r["subject_id"]] = r

    if not args.metrics_only:
        if not args.i_confirm_data_transfer_approved:
            sys.exit("Refusing to send D3TEC transcripts without --i-confirm-data-transfer-approved.")
        (out / "run_info.json").write_text(json.dumps(run_info, indent=1))
        for t in targets:
            if t["subject_id"] in done:
                continue
            prompt = target_block(t)
            t0 = time.time()
            res = call_claude_cli(SYSTEM_PROMPT, prompt, args.model, args.effort, args.timeout)
            answer = res.get("result", "")
            pred, explanation, items = parse_items_answer(answer)
            rec = {"subject_id": t["subject_id"], "label": t["label"], "phq9_score": t["phq9_score"], "pred": pred,
                   "phq9_items": items, "phq9_total": sum(items) if items else None,
                   "pred_text": LABEL_TEXT.get(pred, "INVALID"), "explanation": explanation, "raw_answer": answer,
                   "prompt_sha256": sha256_text(prompt), "effort": args.effort, "usage": res.get("usage"),
                   "cost_usd": res.get("total_cost_usd"), "seconds": round(time.time() - t0, 1)}
            with resp_path.open("a") as f:
                f.write(json.dumps(rec) + "\n")
            done[t["subject_id"]] = rec
            print(f"{t['subject_id']}: gold={LABEL_TEXT[t['label']]} pred={rec['pred_text']} "
                  f"total={rec['phq9_total']} ({rec['seconds']}s)", flush=True)

    records = [done[t["subject_id"]] for t in targets if t["subject_id"] in done]
    if records:
        m = {"split": "all_participants (no official split; zero-shot has no training)",
             "aggregation": "subject-level, one answer per subject", **metrics(records)}
        if any(r["phq9_total"] is not None for r in records) and len({r["label"] for r in records}) == 2:
            m.update(score_metrics(records))
        m["complete"] = len(records) == len(targets)
        (out / "metrics.json").write_text(json.dumps(m, indent=1))
        print(json.dumps(m, indent=1))


if __name__ == "__main__":
    main()
