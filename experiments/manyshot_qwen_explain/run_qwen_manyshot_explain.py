"""Qwen3.8-27B many-shot classification with a generated explanation (advisor prompt v3_advisor).

No embeddings and no head: the model reads numbered labelled examples (full participant transcript ->
gold label) and then one target transcript, and generates a JSON answer with the explanation first
and the label last (greedy, thinking off). The label is read from the generated answer; an answer
without a valid label is INVALID and scored as the wrong class.

Protocols (agreed with the user, 2026-11-24):
  daic     OZY-159/162 layout: the official train + dev subjects (142) are the shots, in one fixed
           order (one shared prefix); each official test subject (47) is classified after them.
  turkish  LOPO over the 120 pooled_t17 participants: the target is classified after the other 119
           (one prompt per target; both question sets of a participant always travel together).
           Metrics are pooled over the 120 answers (one confusion matrix), not averaged.

Text = full transcripts from the reference manifests of the frozen Qwen3.8 text runs (no 4000-char
cap): DAIC ``full_participant_transcript``; Turkish = the positive-set and negative-set subject
transcripts built exactly as the pooled text-only reference (``src/data/runtime``). The manifest file
hash is checked against the reference.

Secondary score: after the answer is generated, the mean log-probability of each label string at the
label slot (same answer text before it) is recorded, giving a continuous margin for AUROC. The
reported label is always the generated one.

Usage:
  python experiments/manyshot_qwen_explain/run_qwen_manyshot_explain.py --dataset daic \
      --manifest M --model-dir D --out-dir OUT --plan-only
  ... (without --plan-only on MN5; resumable: finished targets in responses.jsonl are skipped)
  ... --metrics-only
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import platform
import random
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.manyshot_claude import run_claude_manyshot as claude_prompts  # noqa: E402
from experiments.manyshot_qwen_explain.prompt_v3 import (  # noqa: E402
    LABEL_TEXT, LABEL_VALUE_START, PROMPT_VERSION, cited_examples, parse_answer, system_prompt,
    system_with_shots, target_block,
)

# Prompts: v3_advisor = many-shot with explanation (this module's main protocol). zeroshot_phq8 = the
# Claude zero-shot v4 prompt imported unchanged (eight PHQ-8 item scores 0-3, label = sum >= 10), DAIC only,
# so Qwen and Opus see byte-identical text.
PROMPTS = ("v3_advisor", "zeroshot_phq8")

# Reference manifests of the frozen Qwen3.8 text runs (extraction_metadata manifest_file_sha256).
REFERENCE_MANIFEST_SHA256 = {
    "daic": "8d96abc2a411403f0340792c0906cc1d9d9b8bf6c62ab0b07aa081b42894659d",
    "turkish": "a0daf6658f111f113bcf17231a6abbc0b8a472a3d60d9ddf76cd9ff25eb4e089",
}
DAIC_SHOT_SPLITS = ("train", "val")
DAIC_EVAL_SPLIT = "test"
TURKISH_CONDITIONS = (("pos_only_t17", "Positive question set"), ("negative_only_t17", "Negative question set"))
TURKISH_THRESHOLD = 17.0
USER_TURN_MARK = "<|im_start|>user"
CLOSED_THINK = "<think>\n\n</think>\n\n"
ANSWER_ORDER = (1, 0)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_ids(ids: list[int]) -> str:
    return hashlib.sha256(",".join(map(str, ids)).encode()).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# --------------------------------------------------------------------------- data


def load_daic(rows: list[dict]) -> dict[str, dict]:
    subjects: dict[str, dict] = {}
    for row in rows:
        rec = {"subject_id": str(row["subject_id"]), "split": str(row["split_original"]),
               "label": int(row["label"]), "text": str(row["full_participant_transcript"]).strip(),
               "score": None}
        prev = subjects.setdefault(rec["subject_id"], rec)
        if prev != rec:
            raise ValueError(f"DAIC subject {rec['subject_id']}: rows disagree on split/label/text.")
    return subjects


def load_turkish(rows: list[dict]) -> dict[str, dict]:
    from src.data.runtime import _harmonized_subject_transcripts, _ordered_subject_rows  # noqa: PLC0415

    grouped: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        cond = str(row.get("dataset_variant", "")).strip()
        if cond not in dict(TURKISH_CONDITIONS):
            raise ValueError(f"Unexpected Turkish dataset_variant {cond!r}.")
        grouped[str(row["subject_id"])][cond].append(row)
    subjects: dict[str, dict] = {}
    for sid in sorted(grouped):
        by_cond = grouped[sid]
        if set(by_cond) != set(dict(TURKISH_CONDITIONS)):
            raise ValueError(f"Turkish subject {sid} lacks a question set.")
        all_rows = [r for rs in by_cond.values() for r in rs]
        labels = {int(r["label"]) for r in all_rows}
        scores = {float(r["score"]) for r in all_rows}
        if len(labels) != 1 or len(scores) != 1:
            raise ValueError(f"Turkish subject {sid}: inconsistent label or score.")
        label, score = labels.pop(), scores.pop()
        if label != int(score >= TURKISH_THRESHOLD):
            raise ValueError(f"Turkish subject {sid}: label is not score >= {TURKISH_THRESHOLD}.")
        parts = []
        for cond, title in TURKISH_CONDITIONS:
            text = _harmonized_subject_transcripts(_ordered_subject_rows(by_cond[cond]), "turkish").get(sid, "")
            if not text.strip():
                raise ValueError(f"Turkish subject {sid}: empty {cond} transcript.")
            parts.append(f"[{title}]\n{text.strip()}")
        subjects[sid] = {"subject_id": sid, "split": "all", "label": label, "text": "\n\n".join(parts),
                         "score": score}
    return subjects


def load_subjects(dataset: str, manifest: Path) -> dict[str, dict]:
    rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    subjects = load_daic(rows) if dataset == "daic" else load_turkish(rows)
    for rec in subjects.values():
        if not rec["text"]:
            raise ValueError(f"Subject {rec['subject_id']}: empty transcript.")
    return subjects


def build_tasks(dataset: str, subjects: dict[str, dict], shot_seed: int, prompt: str = "v3_advisor",
                targets_split: str = "test") -> list[dict]:
    """One task per target: its shots (fixed global order) and nothing of the target among them."""
    if prompt == "zeroshot_phq8":
        if dataset != "daic":
            raise ValueError("zeroshot_phq8 is DAIC only (PHQ-8).")
        splits = (DAIC_EVAL_SPLIT,) if targets_split == "test" else DAIC_SHOT_SPLITS
        targets = sorted((s for s in subjects.values() if s["split"] in splits), key=lambda s: int(s["subject_id"]))
        return [{"target": t, "shots": []} for t in targets]
    if targets_split != "test":
        raise ValueError("--targets traindev needs the zero-shot prompt: train+dev are the many-shot examples.")
    if dataset == "daic":
        shots = sorted((s for s in subjects.values() if s["split"] in DAIC_SHOT_SPLITS),
                       key=lambda s: s["subject_id"])
        random.Random(shot_seed).shuffle(shots)
        targets = sorted((s for s in subjects.values() if s["split"] == DAIC_EVAL_SPLIT),
                         key=lambda s: int(s["subject_id"]))
        tasks = [{"target": t, "shots": shots} for t in targets]
    else:
        order = sorted(subjects.values(), key=lambda s: s["subject_id"])
        random.Random(shot_seed).shuffle(order)
        targets = sorted(subjects.values(), key=lambda s: s["subject_id"])
        tasks = [{"target": t, "shots": [s for s in order if s["subject_id"] != t["subject_id"]]}
                 for t in targets]
    for task in tasks:
        ids = {s["subject_id"] for s in task["shots"]}
        if task["target"]["subject_id"] in ids:
            raise AssertionError("The target is among its own shots.")
        if dataset == "daic" and ids & {t["subject_id"] for t in targets}:
            raise AssertionError("A DAIC test subject is among the shots.")
    return tasks


# --------------------------------------------------------------------------- prompts


class Renderer:
    def __init__(self, tokenizer):
        self.tok = tokenizer

    def render(self, system: str, user: str) -> str:
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        text = self.tok.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=False,
                                            tokenize=False)
        if not text.endswith(CLOSED_THINK):
            raise ValueError("Generation prompt does not end with the closed thinking block.")
        return text

    def split(self, text: str, verify: bool) -> tuple[list[int], list[int]]:
        cut = text.rfind(USER_TURN_MARK)
        if cut <= 0:
            raise ValueError("User turn not found.")
        prefix = self.tok(text[:cut], add_special_tokens=True)["input_ids"]
        suffix = self.tok(text[cut:], add_special_tokens=False)["input_ids"]
        if verify and self.tok(text, add_special_tokens=True)["input_ids"] != prefix + suffix:
            raise AssertionError("Prefix + suffix tokens differ from the full prompt tokens.")
        return prefix, suffix


def prompt_texts(dataset: str, task: dict, prompt: str = "v3_advisor") -> tuple[str, str]:
    if prompt == "zeroshot_phq8":
        return (claude_prompts.SYSTEM_PROMPT_ZEROSHOT_V4,
                claude_prompts.target_block(task["target"], zero_shot=True))
    return system_with_shots(dataset, task["shots"]), target_block(task["target"])


def plan(dataset: str, tasks: list[dict], renderer: Renderer, prompt: str = "v3_advisor") -> list[dict]:
    """Token ids per task; DAIC tasks share one prefix (asserted)."""
    planned, prefix_cache = [], {}
    for k, task in enumerate(tasks):
        system, user = prompt_texts(dataset, task, prompt)
        text = renderer.render(system, user)
        key = sha256_text(system)
        if key not in prefix_cache:
            prefix_cache.clear()  # LOPO prefixes are used once; keep memory flat
            prefix, suffix = renderer.split(text, verify=True)
            prefix_cache[key] = prefix
        else:
            prefix = prefix_cache[key]
            cut = text.rfind(USER_TURN_MARK)
            suffix = renderer.tok(text[cut:], add_special_tokens=False)["input_ids"]
            if k < 3:
                full = renderer.tok(text, add_special_tokens=True)["input_ids"]
                if full != prefix + suffix:
                    raise AssertionError("Shared prefix does not tokenize as a prefix.")
        planned.append({**task, "system_sha256": key, "prefix_ids": prefix, "suffix_ids": suffix,
                        "prefix_sha256": sha256_ids(prefix)})
    if dataset == "daic" and len({p["prefix_sha256"] for p in planned}) != 1:
        raise AssertionError("DAIC tasks must share one prefix.")
    return planned


# --------------------------------------------------------------------------- model


class QwenGenerator:
    """Chunked prefill + greedy decoding on the Qwen3.8 language model with a reusable prefix cache."""

    def __init__(self, model_dir: Path, chunk: int):
        import torch  # noqa: PLC0415
        from src.model.qwen38_lora import resolve_qwen38_model_class  # noqa: PLC0415

        self.torch = torch
        self.model = resolve_qwen38_model_class(model_dir).from_pretrained(
            str(model_dir), dtype=torch.bfloat16, attn_implementation="sdpa", device_map="auto",
            local_files_only=True)
        self.model.eval()
        self.decoder = self.model.model.language_model
        self.head = self.model.lm_head
        self.device = self.decoder.embed_tokens.weight.device
        self.chunk = chunk

    def run(self, ids: list[int], cache):
        """Feed ids on top of cache; return (cache, hidden states of all fed positions)."""
        torch = self.torch
        hidden = []
        with torch.inference_mode():
            for start in range(0, len(ids), self.chunk):
                piece = torch.tensor([ids[start:start + self.chunk]], dtype=torch.long, device=self.device)
                out = self.decoder(input_ids=piece, past_key_values=cache, use_cache=True, return_dict=True)
                cache = out.past_key_values
                hidden.append(out.last_hidden_state[0])
        return cache, torch.cat(hidden, 0)

    def prefill(self, ids: list[int]):
        cache, _ = self.run(ids, None)
        return cache

    def copy(self, cache):
        return copy.deepcopy(cache)

    def peak_gb(self) -> list[float]:
        cuda = self.torch.cuda
        return [round(cuda.max_memory_allocated(i) / 2**30, 1) for i in range(cuda.device_count())]

    def logprobs(self, hidden):
        torch = self.torch
        with torch.inference_mode():
            logits = self.head(hidden.to(self.head.weight.device)).float()
            return torch.log_softmax(logits, dim=-1)

    def generate(self, cache, suffix: list[int], max_new: int, eos: set[int]) -> tuple[list[int], bool]:
        """Greedy decoding after suffix. Consumes cache (pass a copy)."""
        cache, h = self.run(suffix, cache)
        last = h[-1:]
        out: list[int] = []
        for _ in range(max_new):
            nxt = int(self.logprobs(last)[0].argmax().item())
            out.append(nxt)
            if nxt in eos:
                return out, True
            cache, h = self.run([nxt], cache)
            last = h[-1:]
        return out, False

    def candidate_scores(self, cache, context: list[int], candidates: list[list[int]]) -> list[float]:
        """Mean log-probability of each candidate after context. Consumes cache (pass a copy)."""
        cache, h = self.run(context, cache)
        last = h[-1:]
        scores = []
        for k, cand in enumerate(candidates):
            c = cache if k == len(candidates) - 1 else self.copy(cache)
            _, hc = self.run(cand, c)
            states = self.torch.cat([last, hc[:-1]], 0)
            lp = self.logprobs(states)
            idx = self.torch.tensor(cand, dtype=self.torch.long, device=lp.device)
            scores.append(float(lp.gather(-1, idx[:, None]).mean().item()))
            del c
        return scores


def label_candidates(tok, answer_text: str) -> tuple[str, list[list[int]]] | None:
    """Answer text up to the label value, and each label's tokens after it (None if not found)."""
    matches = list(LABEL_VALUE_START.finditer(answer_text))
    if not matches:
        return None
    before = answer_text[:matches[-1].end()]
    base = tok(before, add_special_tokens=False)["input_ids"]
    cands = []
    for k in ANSWER_ORDER:
        full = tok(before + LABEL_TEXT[k], add_special_tokens=False)["input_ids"]
        if full[:len(base)] != base or len(full) == len(base):
            return None
        cands.append(full[len(base):])
    return before, cands


# --------------------------------------------------------------------------- metrics


def _f1(tp: int, fp: int, fn: int) -> float:
    return 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0


def _auroc(y: list[int], s: list[float]) -> float:
    pos = [v for v, t in zip(s, y) if t == 1]
    neg = [v for v, t in zip(s, y) if t == 0]
    wins = sum(1.0 if p > n else 0.5 if p == n else 0.0 for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def metrics(records: list[dict]) -> dict:
    """Pooled subject-level metrics (no sklearn: the MN5 inference venv may lack it)."""
    y = [r["label"] for r in records]
    pred = [r["pred"] if r["pred"] is not None else 1 - r["label"] for r in records]  # binary_strict
    tp = sum(t == 1 and p == 1 for t, p in zip(y, pred))
    tn = sum(t == 0 and p == 0 for t, p in zip(y, pred))
    fp = sum(t == 0 and p == 1 for t, p in zip(y, pred))
    fn = sum(t == 1 and p == 0 for t, p in zip(y, pred))
    rec_pos = tp / (tp + fn) if tp + fn else 0.0
    rec_neg = tn / (tn + fp) if tn + fp else 0.0
    out = {"n_subjects": len(y), "invalid": sum(r["pred"] is None for r in records),
           "macro_f1": round((_f1(tp, fp, fn) + _f1(tn, fn, fp)) / 2, 4),
           "positive_f1": round(_f1(tp, fp, fn), 4),
           "uar": round((rec_pos + rec_neg) / 2, 4),
           "confusion": {"tp": tp, "fn": fn, "fp": fp, "tn": tn},
           "predicted_positive": int(sum(pred)), "true_positive": int(sum(y))}
    margins = [r.get("label_margin") for r in records]
    if all(m is not None for m in margins) and len(set(y)) == 2:
        out["auroc_label_margin_secondary"] = round(_auroc(y, margins), 4)
    return out


# --------------------------------------------------------------------------- main


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dataset", required=True, choices=("daic", "turkish"))
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--model-dir", required=True, type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--allow-other-manifest", action="store_true",
                   help="Tests only: skip the reference manifest hash check (never reportable).")
    p.add_argument("--shot-seed", type=int, default=0)
    p.add_argument("--prompt", choices=PROMPTS, default="v3_advisor")
    p.add_argument("--targets", choices=("test", "traindev"), default="test",
                   help="traindev: score the 142 official train+dev subjects (zeroshot_phq8 only, for calibration).")
    p.add_argument("--max-new-tokens", type=int, default=768)
    p.add_argument("--prefill-chunk", type=int, default=8192)
    p.add_argument("--limit", type=int, help="Smoke: only the first N targets.")
    p.add_argument("--shard", help="i/n: run only targets with index %% n == i, writing responses.shard<i>of<n>.jsonl "
                                   "(parallel jobs never share a file; metrics read every responses*.jsonl).")
    p.add_argument("--check-uncached", type=int, default=0,
                   help="For the first N targets also decode from a fresh full-prompt prefill and compare.")
    p.add_argument("--check-determinism", type=int, default=0,
                   help="For the first N targets decode twice from the prefix cache and compare.")
    p.add_argument("--no-label-scores", action="store_true")
    p.add_argument("--time-budget-min", type=float, default=0,
                   help="Stop cleanly (before starting a target) when elapsed time since start plus 1.15x the "
                        "slowest target so far would pass this many minutes; rerun with the same --out-dir to "
                        "resume. For short Slurm limits (acc_debug: 2h). 0 = no budget.")
    p.add_argument("--plan-only", action="store_true", help="Build prompts, count tokens; no model.")
    p.add_argument("--metrics-only", action="store_true")
    return p.parse_args(argv)


def parse_shard(text: str | None) -> tuple[int, int] | None:
    if not text:
        return None
    i, n = (int(x) for x in text.split("/"))
    if not 0 <= i < n:
        raise ValueError(f"Bad --shard {text!r}.")
    return i, n


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                              cwd=PROJECT_ROOT).stdout.strip()
    except OSError:
        return ""


_clock = time.monotonic  # module attribute so tests can fake the clock


def main(argv=None, generator_factory=None) -> None:
    started = _clock()
    args = parse_args(argv)
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    shard = parse_shard(args.shard)
    resp_path = out / (f"responses.shard{shard[0]}of{shard[1]}.jsonl" if shard else "responses.jsonl")

    manifest_sha = sha256_file(args.manifest)
    if manifest_sha != REFERENCE_MANIFEST_SHA256[args.dataset] and not args.allow_other_manifest:
        raise SystemExit(f"Manifest sha256 {manifest_sha[:12]} is not the {args.dataset} reference "
                         f"({REFERENCE_MANIFEST_SHA256[args.dataset][:12]}).")
    subjects = load_subjects(args.dataset, args.manifest)
    tasks = build_tasks(args.dataset, subjects, args.shot_seed, args.prompt, args.targets)
    zs_phq8 = args.prompt == "zeroshot_phq8"
    sys_text = claude_prompts.SYSTEM_PROMPT_ZEROSHOT_V4 if zs_phq8 else system_prompt(args.dataset)
    if args.limit:
        tasks = tasks[:args.limit]

    done: dict[str, dict] = {}
    for path in sorted(out.glob("responses*.jsonl")):
        lines = path.read_text().splitlines()
        for n, line in enumerate(lines):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                if n == len(lines) - 1 and path != resp_path:
                    continue  # another shard is writing its last line right now
                raise
            if r["subject_id"] in done:
                raise ValueError(f"{r['subject_id']} answered twice (second in {path.name}).")
            done[r["subject_id"]] = r

    if not args.metrics_only:
        from transformers import AutoTokenizer  # noqa: PLC0415

        tok = AutoTokenizer.from_pretrained(str(args.model_dir), local_files_only=True)
        renderer = Renderer(tok)
        planned = plan(args.dataset, tasks, renderer, args.prompt)
        prompt_tokens = [len(p["prefix_ids"]) + len(p["suffix_ids"]) for p in planned]
        info = {
            "experiment": "Qwen3.8-27B many-shot with generated explanation (advisor prompt)",
            "dataset": args.dataset,
            "protocol": (f"zero-shot PHQ-8 item scores (label = sum >= 10), official {args.targets} subjects"
                         if zs_phq8 else
                         "official train+dev shots (fixed order), official test targets" if args.dataset == "daic"
                         else "LOPO: each participant classified after the other 119"),
            "prompt_version": "claude_zeroshot_v4" if zs_phq8 else PROMPT_VERSION, "system_prompt": sys_text,
            "system_prompt_sha256": sha256_text(sys_text), "targets": args.targets,
            "manifest": str(args.manifest.resolve()), "manifest_sha256": manifest_sha,
            "manifest_is_reference": manifest_sha == REFERENCE_MANIFEST_SHA256[args.dataset],
            "model_dir": str(args.model_dir), "shot_seed": args.shot_seed,
            "decoding": {"greedy": True, "enable_thinking": False, "max_new_tokens": args.max_new_tokens},
            "prefill_chunk": args.prefill_chunk, "n_subjects": len(subjects), "n_targets": len(planned),
            "n_shots": {"min": min(len(p["shots"]) for p in planned),
                        "max": max(len(p["shots"]) for p in planned)},
            "label_counts": {LABEL_TEXT[k]: sum(s["label"] == k for s in subjects.values()) for k in (1, 0)},
            "transcript_chars": {"max": max(len(s["text"]) for s in subjects.values()),
                                 "n_exactly_4000": sum(len(s["text"]) == 4000 for s in subjects.values())},
            "prompt_tokens": {"min": min(prompt_tokens), "max": max(prompt_tokens)},
            "prefix_sha256_unique": len({p["prefix_sha256"] for p in planned}),
            "git_commit": git_commit(),
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "versions": {"python": platform.python_version()},
        }
        if args.dataset == "daic" and not zs_phq8:
            info["example_subjects"] = [s["subject_id"] for s in planned[0]["shots"]]
        print("PLAN", json.dumps({k: v for k, v in info.items() if k not in ("system_prompt", "example_subjects")}),
              flush=True)
        info_name = "plan.json" if args.plan_only else (f"run_info.shard{shard[0]}of{shard[1]}.json" if shard else "run_info.json")
        info["shard"] = args.shard
        (out / info_name).write_text(json.dumps(info, indent=1))
        if args.plan_only:
            return

        import transformers  # noqa: PLC0415

        info["versions"]["transformers"] = transformers.__version__
        (out / info_name).write_text(json.dumps(info, indent=1))
        gen = (generator_factory or QwenGenerator)(args.model_dir, args.prefill_chunk)
        eos = {i for i in (tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id) if isinstance(i, int)}
        checks = []
        prefix_key, prefix_cache = None, None
        slowest = None
        for k, p in enumerate(planned):
            sid = p["target"]["subject_id"]
            if sid in done or (shard and k % shard[1] != shard[0]):
                continue
            if args.time_budget_min and slowest is not None and \
                    _clock() - started + 1.15 * slowest > args.time_budget_min * 60:
                print(f"TIME BUDGET: stopping before {sid} ({(_clock() - started) / 60:.1f} of "
                      f"{args.time_budget_min} min used, slowest target {slowest / 60:.1f} min); "
                      f"rerun with the same --out-dir to resume.", flush=True)
                break
            t_start = _clock()
            t0 = time.time()
            if p["prefix_sha256"] != prefix_key:
                prefix_cache = None
                prefix_cache = gen.prefill(p["prefix_ids"])
                prefix_key = p["prefix_sha256"]
            t_prefix = time.time() - t0
            gen_ids, finished = gen.generate(gen.copy(prefix_cache), p["suffix_ids"], args.max_new_tokens, eos)
            t_gen = time.time() - t0 - t_prefix
            answer = tok.decode([i for i in gen_ids if i not in eos], skip_special_tokens=False)
            items = None
            if zs_phq8:
                pred, explanation, items = claude_prompts.parse_items_answer(answer)
            else:
                pred, explanation = parse_answer(answer)
            margin, scores = None, None
            if not args.no_label_scores and pred is not None and not zs_phq8:
                lc = label_candidates(tok, answer)
                if lc is not None:
                    before_ids = tok(lc[0], add_special_tokens=False)["input_ids"]
                    sc = gen.candidate_scores(gen.copy(prefix_cache), p["suffix_ids"] + before_ids, lc[1])
                    scores = {LABEL_TEXT[c]: s for c, s in zip(ANSWER_ORDER, sc)}
                    margin = sc[0] - sc[1]
            check = {}
            if k < args.check_determinism:
                again, _ = gen.generate(gen.copy(prefix_cache), p["suffix_ids"], args.max_new_tokens, eos)
                check["determinism_identical"] = again == gen_ids
            if k < args.check_uncached:
                fresh = gen.prefill(p["prefix_ids"])
                ref, _ = gen.generate(fresh, p["suffix_ids"], args.max_new_tokens, eos)
                del fresh
                same = next((j for j, (a, b) in enumerate(zip(ref, gen_ids)) if a != b), None)
                check["uncached_identical"] = ref == gen_ids
                check["uncached_first_divergence"] = same
            if check:
                checks.append({"subject_id": sid, **check})
                print("CHECK", json.dumps(checks[-1]), flush=True)
            shot_ids = [s["subject_id"] for s in p["shots"]]
            cites = [] if zs_phq8 else cited_examples(explanation)
            rec = {"subject_id": sid, "label": p["target"]["label"], "score": p["target"]["score"],
                   "pred": pred, "pred_text": LABEL_TEXT.get(pred, "INVALID"), "explanation": explanation,
                   "raw_answer": answer, "finished": finished, "n_generated_tokens": len(gen_ids),
                   "label_logprob": scores, "label_margin": margin,
                   "cited_examples": [{"number": n, "subject_id": shot_ids[n - 1] if 0 < n <= len(shot_ids) else None,
                                       "label": (LABEL_TEXT[p["shots"][n - 1]["label"]]
                                                 if 0 < n <= len(shot_ids) else None)} for n in cites],
                   "prompt_tokens": len(p["prefix_ids"]) + len(p["suffix_ids"]),
                   "prefix_sha256": p["prefix_sha256"], "suffix_sha256": sha256_ids(p["suffix_ids"]),
                   "seconds": {"prefix": round(t_prefix, 1), "generate": round(t_gen, 1),
                               "total": round(time.time() - t0, 1)}}
            if hasattr(gen, "peak_gb"):
                rec["peak_gpu_gb"] = gen.peak_gb()
            if zs_phq8:
                rec["phq8_items"], rec["phq8_total"] = items, (sum(items) if items is not None else None)
            if args.dataset == "turkish":
                rec["example_subjects"] = shot_ids
            with resp_path.open("a") as f:
                f.write(json.dumps(rec) + "\n")
            done[sid] = rec
            slowest = max(slowest or 0.0, _clock() - t_start)
            print(f"[{k + 1}/{len(planned)}] {sid}: gold={LABEL_TEXT[rec['label']]} pred={rec['pred_text']} "
                  f"tokens={rec['prompt_tokens']}+{len(gen_ids)} {rec['seconds']}", flush=True)
        if checks:
            with (out / (f"checks.shard{shard[0]}of{shard[1]}.jsonl" if shard else "checks.jsonl")).open("a") as f:
                for c in checks:
                    f.write(json.dumps(c) + "\n")

    targets = [t["target"]["subject_id"] for t in tasks]
    records = [done[s] for s in targets if s in done]
    if records and not (shard and len(records) < len(targets)):
        m = {"dataset": args.dataset,
             "split": (f"official_{args.targets}" if args.dataset == "daic" else "LOPO, all participants"),
             "aggregation": "pooled subject-level, one generated answer per subject (INVALID = wrong)",
             **metrics(records), "complete": len(records) == len(targets),
             "unfinished_answers": sum(not r["finished"] for r in records)}
        (out / "metrics.json").write_text(json.dumps(m, indent=1))
        print(json.dumps(m, indent=1))


if __name__ == "__main__":
    main()
