"""DAIC many-shot prompting with Claude Opus as the classifier (OZY-162).

No embeddings and no head: the prompt holds every official train + dev DAIC subject as a labelled
example (participant transcript -> gold label), then one official test subject. Claude answers with
the label and a short explanation grounded in the transcripts. Test subjects never appear among the
shots (asserted).

Text = ``full_participant_transcript`` of the DAIC participant-speech manifest, the same per-subject
text the Qwen3.8 text_only_native runs and the OZY-148/159 many-shot extractor see. The shot block is
identical for every test subject (fixed order), so it is a cacheable prefix.

Backends:
  dry-run     build every prompt, write prompts + size report; sends nothing anywhere.
  claude-cli  ``claude -p`` (the user's Claude subscription). Sends DAIC transcripts to Anthropic.

Zero-shot (``--n-shots 0``): no labelled examples; the system prompt is only the instructions and the
explanation can cite only the participant's transcript.

Usage:
  python experiments/manyshot_claude/run_claude_manyshot.py --out-dir OUT --backend dry-run
  python experiments/manyshot_claude/run_claude_manyshot.py --out-dir OUT --backend claude-cli \
      --i-confirm-data-transfer-approved
Resumable: subjects already in ``responses.jsonl`` are skipped. ``--metrics-only`` recomputes metrics.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_MANIFEST = Path("outputs/manifests_daic_participant_packed30/daic_manifest.jsonl")
LABEL_TEXT = {1: "Depressed", 0: "Non-depressed"}
TEXT_LABEL = {v.lower(): k for k, v in LABEL_TEXT.items()}
SHOT_SPLITS = ("train", "val")  # official train + dev (OZY-159 protocol)
EVAL_SPLIT = "test"

SYSTEM_PROMPT = (
    "You are assisting a research study on depression screening from interview transcripts. "
    "Each transcript contains only the participant's side of a semi-structured clinical interview "
    "(DAIC-WOZ). Labels come from the PHQ-8 questionnaire: Depressed means PHQ-8 >= 10.\n\n"
    "You will first see labelled example participants from this study. Use them to learn how the "
    "labels relate to the transcripts in this dataset. Then classify the final participant.\n\n"
    "Answer with exactly one JSON object and nothing else:\n"
    '{"label": "Depressed" or "Non-depressed", "explanation": "<3-6 sentences>"}\n'
    "The explanation must cite concrete evidence from the final participant's transcript and, where "
    "useful, how it compares with the labelled examples."
)


# v2 (Cenk Hoca, 2026-11-20: "explaining by looking at the examples is important"): the explanation must
# name the most similar labelled examples by number. v1 is kept unchanged for the first reported run.
SYSTEM_PROMPT_V2 = SYSTEM_PROMPT.split("The explanation must")[0] + (
    "The explanation must be grounded in the labelled examples. Name the 2-3 labelled examples that are "
    "most similar to the final participant, by example number and label (for example 'Example 12 "
    "(Depressed)'), and say what they share with the final participant. Cite concrete evidence from the "
    "final participant's transcript. If the most similar examples carry the other label, say why you "
    "still chose this label."
)
PROMPTS = {"v1": SYSTEM_PROMPT, "v2": SYSTEM_PROMPT_V2}

# Zero-shot (Cenk Hoca, 2026-11-23: "Claude opus 5.5 zero shot performans nasil?"): same task, label
# definition and answer format as v2, but no labelled examples, so the explanation can only cite the transcript.
SYSTEM_PROMPT_ZEROSHOT = SYSTEM_PROMPT.split("You will first see")[0] + (
    "Classify the participant whose transcript follows.\n\n"
    "Answer with exactly one JSON object and nothing else:\n"
    '{"label": "Depressed" or "Non-depressed", "explanation": "<3-6 sentences>"}\n'
    "The explanation must cite concrete evidence from the participant's transcript."
)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_subjects(manifest: Path) -> dict[str, dict]:
    """One record per subject: split, label, participant transcript (checked consistent across rows)."""
    subjects: dict[str, dict] = {}
    for line in manifest.read_text().splitlines():
        row = json.loads(line)
        rec = {"subject_id": str(row["subject_id"]), "split": row["split_original"],
               "label": int(row["label"]), "text": row["full_participant_transcript"].strip(),
               "text_sha256": row["full_participant_transcript_sha256"]}
        prev = subjects.setdefault(rec["subject_id"], rec)
        if prev != rec:
            raise ValueError(f"Subject {rec['subject_id']}: rows disagree on split/label/text.")
    for rec in subjects.values():
        if not rec["text"]:
            raise ValueError(f"Subject {rec['subject_id']}: empty transcript.")
    return subjects


def shot_block(shots: list[dict]) -> str:
    parts = ["LABELLED EXAMPLES"]
    for i, s in enumerate(shots, 1):
        parts.append(f"### Example {i}\nTranscript:\n{s['text']}\nLabel: {LABEL_TEXT[s['label']]}")
    return "\n\n".join(parts)


def target_block(target: dict, zero_shot: bool = False) -> str:
    head = "### Participant (classify this one)" if zero_shot else "### Final participant (classify this one)"
    return head + "\nTranscript:\n" + target["text"] + "\n\nReturn the JSON object now."


def parse_answer(text: str) -> tuple[int | None, str]:
    """Label from the last JSON object in the answer; None = INVALID (counted wrong)."""
    for m in reversed(list(re.finditer(r"\{.*?\}", text, flags=re.S))):
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
        label = TEXT_LABEL.get(str(obj.get("label", "")).strip().lower())
        if label is not None:
            return label, str(obj.get("explanation", "")).strip()
    return None, ""


def call_claude_cli(system: str, prompt: str, model: str, effort: str, timeout: int) -> dict:
    """One ``claude -p`` call with no tools, no MCP and no user/project settings or memory files.

    The shots live in the system prompt so the CLI's system-prompt cache breakpoint covers them and
    every test subject after the first reads them from the prompt cache.
    """
    with tempfile.TemporaryDirectory() as cwd:
        sys_file = Path(cwd) / "system.txt"
        sys_file.write_text(system)
        cmd = ["claude", "-p", "--model", model, "--effort", effort, "--output-format", "json",
               "--system-prompt-file", str(sys_file), "--tools", "", "--strict-mcp-config",
               "--setting-sources", "local", "--no-session-persistence"]
        env = {k: v for k, v in os.environ.items() if k != "CLAUDE_EFFORT"}  # effort comes only from --effort
        proc = subprocess.run(cmd, input=prompt, capture_output=True, text=True, cwd=cwd, timeout=timeout, env=env)
    if proc.returncode != 0:
        raise RuntimeError(f"claude exited {proc.returncode}: {proc.stderr[-2000:]}")
    out = json.loads(proc.stdout)
    if out.get("is_error"):
        raise RuntimeError(f"claude returned an error: {str(out)[:2000]}")
    return out


def metrics(records: list[dict]) -> dict:
    from sklearn.metrics import f1_score, recall_score  # noqa: PLC0415

    y = [r["label"] for r in records]
    # binary_strict: INVALID is scored as the wrong class.
    pred = [r["pred"] if r["pred"] is not None else 1 - r["label"] for r in records]
    return {"n_subjects": len(y), "invalid": sum(r["pred"] is None for r in records),
            "macro_f1": round(f1_score(y, pred, average="macro"), 4),
            "positive_f1": round(f1_score(y, pred, pos_label=1, zero_division=0), 4),
            "uar": round(recall_score(y, pred, average="macro"), 4),
            "predicted_positive": int(sum(pred)), "true_positive": int(sum(y))}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    p.add_argument("--backend", choices=("dry-run", "claude-cli"), default="dry-run")
    p.add_argument("--model", default="claude-opus-5-5")
    p.add_argument("--effort", choices=("low", "medium", "high", "xhigh", "max"), default="medium")
    p.add_argument("--prompt-version", choices=sorted(PROMPTS), default="v2")
    p.add_argument("--shot-seed", type=int, default=0)
    p.add_argument("--n-shots", type=int, choices=(0,),
                   help="0 = zero-shot: no labelled examples, zero-shot prompt (ignores --prompt-version). "
                        "Omit for many-shot with every train+dev subject.")
    p.add_argument("--limit", type=int, help="Smoke: only the first N test subjects.")
    p.add_argument("--timeout", type=int, default=900)
    p.add_argument("--i-confirm-data-transfer-approved", action="store_true",
                   help="Required for claude-cli: DAIC-WOZ transcripts leave this machine.")
    p.add_argument("--metrics-only", action="store_true")
    args = p.parse_args()

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    resp_path = out / "responses.jsonl"

    subjects = load_subjects(args.manifest)
    shots = sorted((s for s in subjects.values() if s["split"] in SHOT_SPLITS), key=lambda s: s["subject_id"])
    random.Random(args.shot_seed).shuffle(shots)
    targets = sorted((s for s in subjects.values() if s["split"] == EVAL_SPLIT), key=lambda s: int(s["subject_id"]))
    if {s["subject_id"] for s in shots} & {t["subject_id"] for t in targets}:
        raise AssertionError("A test subject is among the shots.")
    if args.limit:
        targets = targets[:args.limit]
    zero_shot = args.n_shots == 0
    if zero_shot:
        shots, prefix = [], ""
        prompt_version, instructions = "zeroshot", SYSTEM_PROMPT_ZEROSHOT
        system = instructions
    else:
        prefix = shot_block(shots)
        prompt_version, instructions = args.prompt_version, PROMPTS[args.prompt_version]
        system = instructions + "\n\n" + prefix

    run_info = {
        "experiment": ("Claude Opus zero-shot DAIC (no shots)" if zero_shot else "OZY-162 DAIC many-shot prompting with Claude"), "backend": args.backend,
        "model": args.model, "effort": args.effort, "shot_seed": args.shot_seed, "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256_file(args.manifest), "prompt_version": prompt_version, "system_prompt_sha256": sha256_text(instructions),
        "shot_block_sha256": sha256_text(prefix), "system_with_shots_sha256": sha256_text(system),
        "layout": ("system = instructions only; user = target transcript" if zero_shot
                   else "system = instructions + labelled shots; user = target transcript"), "n_shots": len(shots),
        "shot_label_counts": {LABEL_TEXT[k]: sum(s["label"] == k for s in shots) for k in (1, 0)},
        "n_targets": len(targets), "shot_block_chars": len(prefix),
        "approx_shot_tokens": len(prefix) // 4,
        "max_target_chars": max(len(target_block(t, zero_shot)) for t in targets),
        "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    print(json.dumps(run_info, indent=1), flush=True)

    if args.backend == "dry-run" and not args.metrics_only:
        (out / "system_prompt.txt").write_text(instructions)
        (out / "shot_block.txt").write_text(prefix)
        (out / "dry_run.json").write_text(json.dumps(run_info, indent=1))
        print(f"[dry-run] wrote prompts to {out}; nothing was sent.")
        return

    done = {}
    if resp_path.exists():
        for line in resp_path.read_text().splitlines():
            r = json.loads(line)
            done[r["subject_id"]] = r

    if not args.metrics_only:
        if not args.i_confirm_data_transfer_approved:
            sys.exit("Refusing to send DAIC transcripts: pass --i-confirm-data-transfer-approved "
                     "once the DAIC-WOZ data transfer is approved.")
        (out / "run_info.json").write_text(json.dumps(run_info, indent=1))
        for t in targets:
            if t["subject_id"] in done:
                continue
            prompt = target_block(t, zero_shot)
            t0 = time.time()
            res = call_claude_cli(system, prompt, args.model, args.effort, args.timeout)
            answer = res.get("result", "")
            pred, explanation = parse_answer(answer)
            rec = {"subject_id": t["subject_id"], "label": t["label"], "pred": pred,
                   "pred_text": LABEL_TEXT.get(pred, "INVALID"), "explanation": explanation,
                   "raw_answer": answer, "prompt_sha256": sha256_text(prompt),
                   "effort": args.effort, "usage": res.get("usage"), "cost_usd": res.get("total_cost_usd"),
                   "model_usage": res.get("modelUsage"), "seconds": round(time.time() - t0, 1)}
            with resp_path.open("a") as f:
                f.write(json.dumps(rec) + "\n")
            done[t["subject_id"]] = rec
            print(f"{t['subject_id']}: gold={LABEL_TEXT[t['label']]} pred={rec['pred_text']} "
                  f"({rec['seconds']}s)", flush=True)

    records = [done[t["subject_id"]] for t in targets if t["subject_id"] in done]
    if records:
        m = {"split": "official_test", "aggregation": "subject-level, one answer per subject",
             **metrics(records), "complete": len(records) == len(targets)}
        (out / "metrics.json").write_text(json.dumps(m, indent=1))
        print(json.dumps(m, indent=1))


if __name__ == "__main__":
    main()
