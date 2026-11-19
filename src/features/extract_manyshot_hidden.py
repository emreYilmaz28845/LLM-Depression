"""Many-shot-context Qwen3.8-27B text features with leak-free inner cross-fitting.

The frozen text features (``extract_qwen_hidden.py --base-only``) embed each
prompt alone: final post-norm hidden state of the last prompt token. Here the
same prompt follows a block of labelled examples from other training subjects
(user turn = their stored prompt, assistant turn = their gold label), so the
vector is read after the model has seen how labels relate to transcripts in
this study. The examples, rows, partitions and prompts come from the same
reference runs as the frozen Qwen3.8 caches, so the arm without examples
(``--mode zeroshot``) reproduces those caches.

Leakage rule. Per outer fold the fit subjects (``train_subject_ids``) are split
into ``G`` label-stratified inner groups. Shot set ``S_g`` = rows of every fit
subject outside group ``g``.

* A fit subject in group ``g`` is embedded once, with ``S_g``.
* Every other subject the head touches (inner selection, final evaluation,
  DAIC official dev/test) is embedded with each ``S_1..S_G`` and the mean
  vector is stored. The head is linear after a fixed standardization, so this
  equals averaging its logits over the G shot sets.
* Shots never contain the embedded subject or any non-fit subject; both are
  asserted for every forward pass.

Cost. All rows embedded with ``S_g`` share one token prefix (system turn plus
the shots). The prefix is run once, in chunks, and its cache (key/value states
of the 16 full-attention layers and the recurrent and convolution states of the
48 gated-delta-net layers) is copied for each row; only the row's own turn is
run on top. ``--check-uncached`` compares that path with a single cache-free
forward of the whole prompt.

Output: one cache per outer fold in the frozen-cache format
(``outer_train``/``final_eval`` npz + rows + ``extraction_metadata.json``), plus a
``dev`` cache for DAIC. ``baselines/frozen_standalone_heads.py --per-fold-caches``
reads them unchanged.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import platform
import random
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

SCHEMA_VERSION = "manyshot_hidden_cache.v1"
WEIGHTS_SOURCE = {"manyshot": "base_model_manyshot_context", "zeroshot": "base_model_no_adapter"}
POOLING = "last_valid_prompt_token"
HIDDEN_SIZE = 5120
LABEL_TEXT = {1: "Depressed", 0: "Non-depressed"}
# Same note as the colleague's many-shot answer run (manyshot_eval.py), so arm B and arm C see one prompt.
NOTE = ("\n\nThe conversation contains labelled examples: transcripts of other participants of the same "
        "study, each followed by that participant's true study label. Use them as reference for how the "
        "labels relate to the transcripts in this study, then label the final participant.")
USER_TURN_MARK = "<|im_start|>user"
# enable_thinking=False appends a closed empty thinking block after the assistant header. The
# frozen Qwen3.8 caches (arm A) were extracted from prompts that end at the assistant header, so
# the block is removed: both arms read the same position.
CLOSED_THINK = "<think>\n\n</think>\n\n"
OPTIONAL_ROW_KEYS = (
    "response_id", "question_condition", "turn_key", "question_id", "window_id",
    "prompt_id", "segment_index", "num_segments", "chunk_id", "protocol_id", "chunk_index", "num_chunks",
)
PROTOCOLS = ("daic_official", "train_val_test", "train_val")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--routes", required=True, type=Path,
                   help="JSON list of {name, dataset, protocol, reference_checkpoint_dir, "
                        "dev_reference_checkpoint_dir?, split_files, output_dir?}.")
    p.add_argument("--model-dir", required=True, type=Path)
    p.add_argument("--output-root", type=Path, help="Base for routes without an absolute output_dir.")
    p.add_argument("--mode", choices=("manyshot", "zeroshot"), default="manyshot")
    p.add_argument("--folds", type=int, nargs="+", help="Outer folds to run (default: all).")
    p.add_argument("--shot-scheme", choices=("groups", "loo"), default="groups",
                   help="groups: G inner groups (fit rows see the other groups' shots, other rows the mean over "
                        "the G sets). loo: one fixed shot order of all fit subjects; a fit subject is embedded "
                        "with every other fit subject, other rows with all fit subjects.")
    p.add_argument("--answer-scores", action="store_true",
                   help="Also store the mean token log-probability of each label after the prompt "
                        "(the candidate-label likelihood of src/evaluate.score_candidate_label).")
    p.add_argument("--shard", help="i/n: run only shot sets with index %% n == i. Finished sets are saved "
                                   "and skipped on restart; the cache is written by a run without --shard.")
    p.add_argument("--only-sets", type=int, nargs="+",
                   help="Smoke only: run these shot-set indices (negative = from the end); no cache written.")
    p.add_argument("--inner-groups", type=int, default=5)
    p.add_argument("--inner-seed", type=int, default=1337)
    p.add_argument("--shot-seed", type=int, default=0)
    p.add_argument("--prefill-chunk", type=int, default=8192)
    p.add_argument("--check-uncached", type=int, default=0,
                   help="Rows of the first shot set to recompute with one cache-free forward.")
    p.add_argument("--compare-cache", type=Path,
                   help="Zero-shot check: frozen cache dir whose vectors must match by sample_id.")
    p.add_argument("--max-shot-sets", type=int, help="Smoke only: stop after this many shot sets per fold.")
    p.add_argument("--plan-only", action="store_true", help="Build prompts and report token counts; no model.")
    return p.parse_args()


# --------------------------------------------------------------------------- data


def load_examples(ref_dir: Path, final_eval_source: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """All examples of one reference run (outer_train + final_eval), with provenance.

    Mirrors ``extract_deepseek_hidden.build_route`` up to the examples; the prompt
    texts are the stored ``prompt_system_text`` / ``prompt_user_text``.
    """
    from src.data.emotion import use_emotion  # noqa: PLC0415
    from src.data.runtime import load_manifest_rows  # noqa: PLC0415
    from src.features.extract_qwen_hidden import (  # noqa: PLC0415
        _load_saved_run, _partition_examples, _resolve_subject_partitions, _saved_path,
        _validate_saved_split, ensure_hidden_extraction_supported, resolve_condition,
    )
    from src.utils import read_json, sha256_file, sha256_jsonl_rows  # noqa: PLC0415

    ref = ref_dir.resolve()
    saved, config, run_config_path, split_path = _load_saved_run(ref)
    ensure_hidden_extraction_supported(config, run_config_path)
    config = json.loads(json.dumps(config))
    fold = int(saved["fold"])
    split_payload = read_json(split_path)
    partition_ids, evaluation_provenance = _resolve_subject_partitions(
        saved, config, split_payload, final_eval_source=final_eval_source)
    cv_protocol = str(saved.get("cv_protocol") or config.get("split", {}).get("cv_protocol") or "")
    split_metadata_path = _validate_saved_split(
        saved, config, partition_ids, fold, 1 if cv_protocol == "train_val" else 2,
        split_payload=split_payload, final_eval_source=final_eval_source)
    manifest_path = _saved_path(saved["manifest_path"])
    manifest_rows = load_manifest_rows(manifest_path)
    manifest_hash = sha256_jsonl_rows(manifest_rows)
    if saved.get("manifest_hash") and manifest_hash != saved["manifest_hash"]:
        raise ValueError("Manifest hash differs from the reference run.")
    if saved.get("input_modality") not in (None, "text_only", "text"):
        raise ValueError(f"Text-only extraction; got {saved.get('input_modality')!r}.")
    examples, _ = _partition_examples(manifest_rows, config, partition_ids, fold, checkpoint_dir=ref)
    out = []
    for partition in ("outer_train", "final_eval"):
        for ex in examples[partition]:
            user = ex.get("prompt_user_text")
            if not isinstance(user, str) or not user.strip():
                raise ValueError(f"{ex.get('sample_id')}: missing prompt_user_text.")
            row = {"dataset": str(ex["dataset"]), "sample_id": str(ex["sample_id"]),
                   "subject_id": str(ex["subject_id"]), "label": int(ex["label"])}
            row.update({k: ex[k] for k in OPTIONAL_ROW_KEYS if k in ex})
            out.append({"row": row, "system": ex.get("prompt_system_text") or "", "user": user,
                        "reference_prompt_text": None})
    provenance = {
        "dataset": config["dataset"],
        "condition": resolve_condition(None, saved.get("input_modality"), use_emotion(config)),
        "input_modality": saved.get("input_modality"),
        "protocol_id": config.get("protocol_id", ""),
        "reference_checkpoint_dir": str(ref),
        "saved_run_config": str(run_config_path), "saved_run_config_sha256": sha256_file(run_config_path),
        "saved_split": str(split_path), "saved_split_sha256": sha256_file(split_path),
        "split_metadata": str(split_metadata_path), "split_metadata_sha256": sha256_file(split_metadata_path),
        "manifest": str(manifest_path), "manifest_sha256": manifest_hash,
        "manifest_file_sha256": sha256_file(manifest_path),
        "evaluation_provenance": evaluation_provenance, "final_eval_source": final_eval_source,
    }
    return out, provenance


def build_pool(route: dict[str, Any]) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Rows of every subject in the route, by subject (DAIC adds its official dev subjects)."""
    examples, prov = load_examples(Path(route["reference_checkpoint_dir"]), "saved")
    if route.get("dev_reference_checkpoint_dir"):
        dev, dev_prov = load_examples(Path(route["dev_reference_checkpoint_dir"]), "selection")
        if dev_prov["manifest_sha256"] != prov["manifest_sha256"]:
            raise ValueError("Dev reference run uses a different manifest.")
        seen = {e["row"]["sample_id"] for e in examples}
        examples += [e for e in dev if e["row"]["sample_id"] not in seen]
        prov["dev_reference"] = {k: dev_prov[k] for k in ("reference_checkpoint_dir", "saved_split_sha256")}
    pool: dict[str, list[dict[str, Any]]] = {}
    for e in examples:
        pool.setdefault(e["row"]["subject_id"], []).append(e)
    if len({e["row"]["sample_id"] for e in examples}) != len(examples):
        raise ValueError("Duplicate sample ids in the pool.")
    systems = {e["system"] for e in examples}
    if len(systems) != 1:
        raise ValueError(f"{route['name']}: expected one system prompt, got {len(systems)}.")
    for subject, rows in pool.items():
        if len({r["row"]["label"] for r in rows}) != 1:
            raise ValueError(f"Subject {subject} has conflicting labels.")
        rows.sort(key=lambda r: r["row"]["sample_id"])
    return pool, prov


def fold_plan(route: dict[str, Any], split: dict[str, Any], pool: dict[str, Any]) -> dict[str, list[str]]:
    """Which subjects are fit (shot donors), which go to which cache partition."""
    protocol = route["protocol"]
    if protocol not in PROTOCOLS:
        raise ValueError(f"Unknown protocol {protocol!r}.")
    fit = sorted(map(str, split["train_subject_ids"]))
    sel = sorted(map(str, split.get("selection_subject_ids") or []))
    fin = sorted(map(str, split.get("final_eval_subject_ids") or []))
    if protocol == "train_val":
        if fin:
            raise ValueError("train_val split has final_eval subjects.")
        parts = {"outer_train": fit, "final_eval": sel}
    elif protocol == "train_val_test":
        parts = {"outer_train": fit + sel, "final_eval": fin}
    else:
        parts = {"outer_train": fit, "final_eval": fin, "dev": sel}
    groups = [set(fit), set(sel), set(fin)]
    if any(a & b for i, a in enumerate(groups) for b in groups[i + 1:]):
        raise ValueError("Fit, selection and final subjects overlap.")
    missing = (set(fit) | set(sel) | set(fin)) - set(pool)
    if missing:
        raise ValueError(f"{len(missing)} split subjects are not in the prompt pool.")
    if not fit or not parts["final_eval"]:
        raise ValueError("Empty fit or report subjects.")
    return {"fit": fit, "others": sorted(set(sel) | set(fin)), **parts}


def inner_groups(fit: list[str], labels: dict[str, int], n: int, seed: int, fold: int) -> list[list[str]]:
    """Label-stratified subject groups: shuffle each class, deal round-robin."""
    rng = np.random.default_rng([seed, fold])
    groups: list[list[str]] = [[] for _ in range(n)]
    position = 0
    for label in (1, 0):
        for subject in rng.permutation([s for s in fit if labels[s] == label]).tolist():
            groups[position % n].append(subject)
            position += 1
    if any(not g for g in groups):
        raise ValueError("An inner group is empty.")
    return groups


def shot_rows(pool, donors: list[str], seed: int) -> list[dict[str, Any]]:
    rows = [r for s in sorted(donors) for r in pool[s]]
    random.Random(seed).shuffle(rows)
    return rows


def messages(system: str, shots: list[dict[str, Any]], user: str) -> list[dict[str, str]]:
    msgs = [{"role": "system", "content": system + (NOTE if shots else "")}] if (system or shots) else []
    for s in shots:
        msgs += [{"role": "user", "content": s["user"]},
                 {"role": "assistant", "content": LABEL_TEXT[s["row"]["label"]]}]
    return msgs + [{"role": "user", "content": user}]


class Renderer:
    """Chat-template rendering and the prefix/suffix token split of one shot set."""

    def __init__(self, tokenizer):
        self.tok = tokenizer

    def text(self, msgs):
        text = self.tok.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=False,
                                            tokenize=False)
        # Kept, as in the reference extractor (render_qwen38_prompt): the pooled token is the
        # last token of the closed thinking block, the exact position the model answers from.
        if not text.endswith(CLOSED_THINK):
            raise ValueError("Generation prompt does not end with the closed thinking block.")
        return text

    @staticmethod
    def cut(text: str) -> int:
        cut = text.rfind(USER_TURN_MARK)
        if cut <= 0:
            raise ValueError("Last user turn not found in the rendered prompt.")
        return cut

    def split(self, text: str) -> tuple[list[int], list[int], str]:
        """Full check: tokenize the whole prompt and the prefix (slow for long prompts)."""
        cut = self.cut(text)
        full = self.tok(text, add_special_tokens=True)["input_ids"]
        prefix = self.tok(text[:cut], add_special_tokens=True)["input_ids"]
        if full[:len(prefix)] != prefix:
            raise ValueError("Prefix tokens are not a prefix of the full prompt tokens.")
        return prefix, full[len(prefix):], text

    def suffix(self, text: str, prefix_text: str, prefix_ids: list[int], verify: bool) -> list[int]:
        """Tokens of the last user turn; the prefix text must match the shot set's prefix."""
        cut = self.cut(text)
        if text[:cut] != prefix_text:
            raise AssertionError("Rows of one shot set do not share the prefix.")
        suffix = self.tok(text[cut:], add_special_tokens=False)["input_ids"]
        if verify and self.tok(text, add_special_tokens=True)["input_ids"] != prefix_ids + suffix:
            raise AssertionError("Prefix + suffix tokens differ from the full prompt tokens.")
        return suffix


def sha(ids: list[int]) -> str:
    return hashlib.sha256(np.asarray(ids, dtype=np.int64).tobytes()).hexdigest()


# --------------------------------------------------------------------------- model


class Encoder:
    def __init__(self, model_dir: Path, chunk: int):
        import torch  # noqa: PLC0415
        from src.model.qwen38_lora import resolve_qwen38_model_class  # noqa: PLC0415

        self.torch = torch
        self.model = resolve_qwen38_model_class(model_dir).from_pretrained(
            str(model_dir), dtype=torch.bfloat16, attn_implementation="sdpa",
            device_map="auto", local_files_only=True)
        self.model.eval()
        if type(self.model).__name__ != "Qwen3_5ForConditionalGeneration":
            raise ValueError(f"Unexpected model class {type(self.model).__name__}.")
        self.decoder = self.model.model.language_model
        self.device = self.decoder.embed_tokens.weight.device
        self.chunk = chunk

    def _run(self, ids: list[int], cache):
        torch = self.torch
        out = None
        for start in range(0, len(ids), self.chunk):
            piece = torch.tensor([ids[start:start + self.chunk]], dtype=torch.long, device=self.device)
            out = self.decoder(input_ids=piece, past_key_values=cache, use_cache=True, return_dict=True)
            cache = out.past_key_values
        return cache, out.last_hidden_state[0, -1].float().cpu().numpy()

    def prefix(self, ids: list[int]):
        with self.torch.inference_mode():
            cache, _ = self._run(ids, None)
        return cache

    def row(self, prefix_cache, suffix: list[int]) -> np.ndarray:
        with self.torch.inference_mode():
            _, vec = self._run(suffix, copy.deepcopy(prefix_cache))
        return vec.astype(np.float32)

    def row_scores(self, prefix_cache, suffix: list[int], candidates: list[list[int]]):
        """Vector of the last prompt token and the mean log-probability of each candidate's tokens.

        One forward of ``suffix + candidate`` per candidate on a copy of the prefix cache; the
        prompt positions are causal, so the vector does not depend on the candidate (checked).
        """
        torch = self.torch
        n = len(suffix)
        vecs, scores = [], []
        with torch.inference_mode():
            for cand in candidates:
                ids, cache, hidden = suffix + cand, copy.deepcopy(prefix_cache), []
                for start in range(0, len(ids), self.chunk):
                    piece = torch.tensor([ids[start:start + self.chunk]], dtype=torch.long, device=self.device)
                    out = self.decoder(input_ids=piece, past_key_values=cache, use_cache=True, return_dict=True)
                    cache = out.past_key_values
                    lo = max(n - 1 - start, 0)
                    if start + piece.shape[1] > n - 1:
                        hidden.append(out.last_hidden_state[0, lo:])
                del cache
                h = torch.cat(hidden, 0)  # positions n-1 .. len(ids)-1
                vecs.append(h[0].float().cpu().numpy().astype(np.float32))
                head = self.model.lm_head
                logits = head(h[:len(cand)].to(head.weight.device)).float()
                logp = torch.log_softmax(logits, dim=-1)
                target = torch.tensor(cand, dtype=torch.long, device=logp.device)
                scores.append(float(logp.gather(-1, target[:, None]).squeeze(-1).mean().item()))
        return vecs[0], scores, float(max(np.abs(v - vecs[0]).max() for v in vecs))

    def uncached(self, ids: list[int]) -> np.ndarray:
        torch = self.torch
        with torch.inference_mode():
            out = self.decoder(input_ids=torch.tensor([ids], dtype=torch.long, device=self.device),
                               use_cache=False, return_dict=True)
        return out.last_hidden_state[0, -1].float().cpu().numpy().astype(np.float32)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


# --------------------------------------------------------------------------- run


def plan_fold(route, split, pool, fold, args, renderer):
    """Shot sets, rows to embed with each, and token counts (CPU only)."""
    plan = fold_plan(route, split, pool)
    labels = {s: pool[s][0]["row"]["label"] for s in pool}
    system = next(iter(pool.values()))[0]["system"]
    if args.mode == "zeroshot":
        sets = [{"group": None, "donors": [], "own": plan["fit"], "shared": plan["others"]}]
    elif getattr(args, "shot_scheme", "groups") == "loo":
        sets = [{"group": s, "donors": [d for d in plan["fit"] if d != s], "own": [s], "shared": []}
                for s in plan["fit"]]
        sets.append({"group": "all", "donors": list(plan["fit"]), "own": [], "shared": plan["others"]})
    else:
        groups = inner_groups(plan["fit"], labels, args.inner_groups, args.inner_seed, fold)
        sets = [{"group": g, "donors": [s for s in plan["fit"] if s not in set(members)],
                 "own": members, "shared": plan["others"]} for g, members in enumerate(groups)]
    loo = args.mode == "manyshot" and getattr(args, "shot_scheme", "groups") == "loo"
    # LOO: one shot order for the fold; each set drops one subject's turns from it, so every
    # context is the same up to that subject (fit and evaluation rows see nearly the same prompt).
    order = shot_rows(pool, plan["fit"], args.shot_seed * 100003 + fold * 101) if loo else None
    for i, st in enumerate(sets):
        if loo:
            keep = set(st["donors"])
            shots = [r for r in order if r["row"]["subject_id"] in keep]
        else:
            shots = shot_rows(pool, st["donors"], args.shot_seed * 100003 + fold * 101 + i)
        donors = set(st["donors"])
        for s in st["own"] + st["shared"]:
            if s in donors:
                raise AssertionError("A subject is among its own shots.")
        if not donors <= set(plan["fit"]):
            raise AssertionError("Shots contain a non-fit subject.")
        first = pool[(st["own"] + st["shared"])[0]][0]
        first_text = renderer.text(messages(system, shots, first["user"]))
        prefix_text = first_text[:renderer.cut(first_text)]
        prefix = renderer.tok(prefix_text, add_special_tokens=True)["input_ids"]
        st.update({"shots": shots, "prefix_ids": prefix, "prefix_sha256": sha(prefix),
                   "n_shot_rows": len(shots), "n_shot_subjects": len(donors)})
        st["rows"] = []
        for s in st["own"] + st["shared"]:
            for r in pool[s]:
                text = renderer.text(messages(system, shots, r["user"]))
                suf = renderer.suffix(text, prefix_text, prefix, verify=len(st["rows"]) < 2)
                if args.mode == "zeroshot" and text != _reference_prompt(renderer, system, r):
                    raise AssertionError(f"{r['row']['sample_id']}: zero-shot prompt differs from the reference.")
                st["rows"].append({"subject": s, "own": s in set(st["own"]), "ex": r, "suffix": suf})
    return plan, sets, system


def _reference_prompt(renderer, system: str, r: dict[str, Any]) -> str:
    """The prompt the frozen Qwen3.8 extractor feeds (prepare_qwen38_example -> render_qwen38_prompt)."""
    if r.get("reference_prompt_text") is not None:
        return r["reference_prompt_text"]
    from src.model.qwen38_lora import render_qwen38_prompt  # noqa: PLC0415

    return render_qwen38_prompt(renderer.tok, system, r["user"])


def write_cache(out_dir: Path, name: str, rows: list[dict], vectors: list[np.ndarray]) -> dict[str, Any]:
    from src.utils import write_jsonl  # noqa: PLC0415

    matrix = np.stack(vectors).astype(np.float32) if vectors else np.empty((0, HIDDEN_SIZE), np.float32)
    np.savez_compressed(out_dir / f"{name}.npz", vectors=matrix)
    write_jsonl(rows, out_dir / f"{name}_rows.jsonl")
    return {"rows": len(rows), "subjects": len({r["subject_id"] for r in rows})}


def main() -> None:
    args = parse_args()
    from transformers import AutoTokenizer  # noqa: PLC0415
    from src.utils import read_json, save_json, sha256_file  # noqa: PLC0415

    renderer = Renderer(AutoTokenizer.from_pretrained(str(args.model_dir), local_files_only=True))
    routes = json.loads(args.routes.read_text())
    jobs = []
    for route in routes:
        pool, prov = build_pool(route)
        folds = args.folds if args.folds is not None else list(range(len(route["split_files"])))
        for fold in folds:
            base = Path(route.get("output_dir") or route["name"])
            if not base.is_absolute():
                if args.output_root is None:
                    raise ValueError("Relative output_dir needs --output-root.")
                base = args.output_root / base
            out_dir = base / f"fold_{fold}"
            if (out_dir / "extraction_metadata.json").exists():
                print(f"[skip] {route['name']} fold {fold}: complete", flush=True)
                continue
            split_path = Path(route["split_files"][fold])
            plan, sets, system = plan_fold(route, read_json(split_path), pool, fold, args, renderer)
            if args.max_shot_sets:
                sets = sets[:args.max_shot_sets]
            summary = {"route": route["name"], "fold": fold, "mode": args.mode, "shot_sets": len(sets),
                       "prefix_tokens": [len(s["prefix_ids"]) for s in sets],
                       "rows_per_set": [len(s["rows"]) for s in sets],
                       "suffix_tokens_max": max(len(r["suffix"]) for s in sets for r in s["rows"]),
                       "fit": len(plan["fit"]), "others": len(plan["others"])}
            print("PLAN", json.dumps(summary), flush=True)
            jobs.append((route, prov, fold, split_path, plan, sets, system, out_dir, summary))
    if args.plan_only or not jobs:
        return
    shard = parse_shard(args.shard)
    candidates = None
    if args.answer_scores:
        candidates = [renderer.tok(LABEL_TEXT[k], add_special_tokens=False)["input_ids"] for k in ANSWER_ORDER]

    if candidates is not None:
        for _r, _p, _f, _sp, _pl, sets, system, _o, _s in jobs:
            first = sets[0]["rows"][0]
            text = renderer.text(messages(system, sets[0]["shots"], first["ex"]["user"]))
            for cand, k in zip(candidates, ANSWER_ORDER):
                full = renderer.tok(text + LABEL_TEXT[k], add_special_tokens=True)["input_ids"]
                if full != sets[0]["prefix_ids"] + first["suffix"] + cand:
                    raise AssertionError(f"Label {LABEL_TEXT[k]!r} does not tokenize the same after the prompt.")
    selected = {j[2]: pick_sets(len(j[5]), args.only_sets) for j in jobs}
    need_model = any(i in selected[j[2]] for j in jobs for i in set_todo(j[7], j[5], shard))
    enc = compare = None
    if need_model:
        import torch  # noqa: PLC0415

        torch.manual_seed(0)
        enc = Encoder(args.model_dir, args.prefill_chunk)
        if args.compare_cache:
            from baselines.subject_mean_logreg_head import load_partition  # noqa: PLC0415
            compare = {}
            for part in ("outer_train", "final_eval"):
                x, rows = load_partition(args.compare_cache, part)
                compare.update({r["sample_id"]: x[i] for i, r in enumerate(rows)})

    for route, prov, fold, split_path, plan, sets, system, out_dir, summary in jobs:
        set_dir = sets_dir(out_dir)
        set_dir.mkdir(parents=True, exist_ok=True)
        for i in set_todo(out_dir, sets, shard):
            if i not in selected[fold]:
                continue
            run_set(enc, sets[i], i, set_dir, args, candidates, compare, route["name"], fold)
        if shard is not None or args.max_shot_sets or args.only_sets:
            done = sum((set_dir / f"set_{i:04d}.json").exists() for i in range(len(sets)))
            print(f"[shard] {route['name']} fold {fold}: {done}/{len(sets)} sets on disk; cache not written",
                  flush=True)
            continue
        assemble(route, prov, fold, split_path, plan, sets, out_dir, args, renderer)


ANSWER_ORDER = (1, 0)  # answer score columns: Depressed, Non-depressed


def pick_sets(n: int, only: list[int] | None) -> set[int]:
    """--only-sets indices (negative counts from the end) or all sets."""
    return set(range(n)) if not only else {i % n for i in only}


def parse_shard(text: str | None) -> tuple[int, int] | None:
    if not text:
        return None
    i, n = (int(x) for x in text.split("/"))
    if not 0 <= i < n:
        raise ValueError(f"Bad --shard {text!r}.")
    return i, n


def sets_dir(out_dir: Path) -> Path:
    return out_dir.with_name(out_dir.name + ".sets")


def set_todo(out_dir: Path, sets: list[dict], shard) -> list[int]:
    """Indices of this shard's shot sets that have no saved result yet."""
    if (out_dir / "extraction_metadata.json").exists():
        return []
    d = sets_dir(out_dir)
    idx = [i for i in range(len(sets)) if shard is None or i % shard[1] == shard[0]]
    return [i for i in idx if not (d / f"set_{i:04d}.json").exists()]


def run_set(enc, st, i, set_dir: Path, args, candidates, compare, route_name, fold) -> None:
    """Embed (and score) every row of one shot set and save the result atomically."""
    import torch  # noqa: PLC0415

    checks: dict[str, Any] = {"uncached": [], "determinism_max_abs_diff": None, "compare_cache": [],
                              "candidate_vector_max_abs_diff": None}
    t0 = time.time()
    cache = enc.prefix(st["prefix_ids"])
    t_prefix = time.time() - t0
    t1 = time.time()
    vecs, scores = [], []
    for j, item in enumerate(st["rows"]):
        ex = item["ex"]
        if candidates is not None:
            vec, sc, cand_diff = enc.row_scores(cache, item["suffix"], candidates)
            scores.append(sc)
            checks["candidate_vector_max_abs_diff"] = max(cand_diff, checks["candidate_vector_max_abs_diff"] or 0.0)
            if j == 0:
                plain = enc.row(cache, item["suffix"])
                checks["scoring_vs_plain_max_abs_diff"] = float(np.abs(vec - plain).max())
        else:
            vec = enc.row(cache, item["suffix"])
        if not np.isfinite(vec).all() or vec.shape != (HIDDEN_SIZE,):
            raise ValueError(f"Bad vector for {ex['row']['sample_id']}.")
        if j == 0:
            again = enc.row(cache, item["suffix"])
            checks["determinism_max_abs_diff"] = float(np.abs(vec - again).max())
        if j < args.check_uncached and i == 0:
            ref = enc.uncached(st["prefix_ids"] + item["suffix"])
            checks["uncached"].append({"max_abs_diff": float(np.abs(vec - ref).max()),
                                       "max_abs_ref": float(np.abs(ref).max()), "cosine": cosine(vec, ref)})
        if compare is not None and ex["row"]["sample_id"] in compare:
            other = compare[ex["row"]["sample_id"]].astype(np.float32)
            checks["compare_cache"].append({"max_abs_diff": float(np.abs(vec - other).max()),
                                            "cosine": cosine(vec, other)})
        vecs.append(vec)
    del cache
    torch.cuda.empty_cache()
    timing = {"set": i, "prefix_tokens": len(st["prefix_ids"]), "prefix_seconds": round(t_prefix, 1),
              "rows": len(st["rows"]), "rows_seconds": round(time.time() - t1, 1)}
    print("SET", json.dumps({"route": route_name, "fold": fold, **timing, "checks": {
        k: v for k, v in checks.items() if not isinstance(v, list)}}), flush=True)
    if checks["uncached"]:
        print("CHECK_UNCACHED", json.dumps(checks["uncached"]), flush=True)
    if checks["compare_cache"]:
        cc = checks["compare_cache"]
        print("CHECK_COMPARE", json.dumps({"n": len(cc), "min_cosine": min(c["cosine"] for c in cc),
                                           "max_abs_diff": max(c["max_abs_diff"] for c in cc)}), flush=True)
    tmp = set_dir / f"set_{i:04d}.tmp.npz"
    np.savez(tmp, vectors=np.stack(vecs).astype(np.float32),
             scores=np.asarray(scores, dtype=np.float64).reshape(len(vecs), -1))
    tmp.rename(set_dir / f"set_{i:04d}.npz")
    meta = {"set": i, "group": st["group"], "prefix_sha256": st["prefix_sha256"],
            "sample_ids": [r["ex"]["row"]["sample_id"] for r in st["rows"]],
            "own": [bool(r["own"]) for r in st["rows"]], "timing": timing, "checks": checks}
    (set_dir / f"set_{i:04d}.tmp.json").write_text(json.dumps(meta))
    (set_dir / f"set_{i:04d}.tmp.json").rename(set_dir / f"set_{i:04d}.json")


def assemble(route, prov, fold, split_path, plan, sets, out_dir: Path, args, renderer) -> None:
    """Build the frozen-format cache from the saved shot-set results of one fold."""
    import transformers  # noqa: PLC0415
    from src.utils import save_json, sha256_file  # noqa: PLC0415

    set_dir = sets_dir(out_dir)
    own_vec: dict[str, np.ndarray] = {}
    own_score: dict[str, np.ndarray] = {}
    shared_sum: dict[str, np.ndarray] = {}
    shared_score: dict[str, np.ndarray] = {}
    shared_n: dict[str, int] = {}
    timing, set_checks = [], []
    for i, st in enumerate(sets):
        meta = json.loads((set_dir / f"set_{i:04d}.json").read_text())
        data = np.load(set_dir / f"set_{i:04d}.npz")
        expected_ids = [r["ex"]["row"]["sample_id"] for r in st["rows"]]
        if meta["prefix_sha256"] != st["prefix_sha256"] or meta["sample_ids"] != expected_ids:
            raise AssertionError(f"Saved set {i} does not match the current plan.")
        timing.append(meta["timing"])
        set_checks.append({"set": i, **meta["checks"]})
        has_scores = data["scores"].shape[1] > 0
        for k, (sid, own) in enumerate(zip(meta["sample_ids"], meta["own"])):
            vec = data["vectors"][k]
            sc = data["scores"][k] if has_scores else None
            if own:
                if sid in own_vec:
                    raise AssertionError(f"{sid} embedded twice as a fit row.")
                own_vec[sid] = vec
                if sc is not None:
                    own_score[sid] = sc
            else:
                shared_sum[sid] = shared_sum.get(sid, 0) + vec.astype(np.float64)
                shared_n[sid] = shared_n.get(sid, 0) + 1
                if sc is not None:
                    shared_score[sid] = shared_score.get(sid, 0) + sc
    expected = sum(1 for st in sets if st["shared"])
    vectors: dict[str, np.ndarray] = dict(own_vec)
    answer: dict[str, np.ndarray] = dict(own_score)
    for sid, total in shared_sum.items():
        if shared_n[sid] != expected:
            raise AssertionError(f"{sid}: {shared_n[sid]} of {expected} shot sets.")
        vectors[sid] = (total / shared_n[sid]).astype(np.float32)
        if sid in shared_score:
            answer[sid] = shared_score[sid] / shared_n[sid]
    partial = out_dir.with_name(out_dir.name + ".partial")
    if partial.exists():
        raise RuntimeError(f"Partial output exists: {partial}")
    partial.mkdir(parents=True)
    pool_rows = {r["ex"]["row"]["sample_id"]: r["ex"]["row"] for st in sets for r in st["rows"]}
    parts = {}
    layout = [("outer_train", partial, "outer_train"), ("final_eval", partial, "final_eval")]
    if "dev" in plan:
        (partial / "dev").mkdir()
        layout.append(("dev", partial / "dev", "final_eval"))
    loo = args.mode == "manyshot" and args.shot_scheme == "loo"
    checks = {"per_set": set_checks,
              "determinism_max_abs_diff": max((c["determinism_max_abs_diff"] or 0.0) for c in set_checks),
              "uncached": [u for c in set_checks for u in c["uncached"]],
              "compare_cache_min_cosine": min((u["cosine"] for c in set_checks for u in c["compare_cache"]),
                                              default=None)}
    meta_common = {
        "schema_version": SCHEMA_VERSION, "model_backend": "qwen38",
        "base_model": "/gpfs/projects/etur92/ozu647717/models/Qwen3.8-27B",
        "base_model_path": str(args.model_dir),
        "checkpoint_type": "base_model", "weights_source": WEIGHTS_SOURCE[args.mode],
        "hidden_layer": "final", "pooling": POOLING, "vector_dimension": HIDDEN_SIZE, "vector_dtype": "float32",
        "hidden_definition": "language-model decoder final RMSNorm output, last prompt token",
        "prompt_encoding": "apply_chat_template(add_generation_prompt=True, enable_thinking=False)",
        "manyshot": {
            "mode": args.mode, "shot_scheme": args.shot_scheme if args.mode == "manyshot" else None,
            "note": NOTE if args.mode == "manyshot" else None,
            "label_text": LABEL_TEXT,
            "inner_groups": args.inner_groups if args.mode == "manyshot" and not loo else 0,
            "inner_seed": args.inner_seed, "shot_seed": args.shot_seed,
            "shot_donors": ("all other fit subjects, one fixed order per fold" if loo else
                            "fit subjects outside the embedded subject's inner group"),
            "non_fit_vectors": "one set with all fit subjects as shots" if loo else "mean over all shot sets",
            "n_shot_sets": len(sets),
            "prefix_sha256": [s["prefix_sha256"] for s in sets],
            "prefix_tokens": [len(s["prefix_ids"]) for s in sets],
            "shot_rows": [s["n_shot_rows"] for s in sets],
            "shot_subjects": [s["n_shot_subjects"] for s in sets],
            "cache_reuse": "prefix cache deep-copied per row; prefill chunk " + str(args.prefill_chunk),
        },
        "answer_scores": ({"file": "<partition>_answer_scores.jsonl", "columns": [LABEL_TEXT[k] for k in ANSWER_ORDER],
                           "score": "mean token log-probability of the label after the prompt "
                                    "(src/evaluate.score_candidate_label)"} if answer else None),
        "outer_fold": fold, "protocol": route["protocol"],
        "split_file": str(split_path), "split_file_sha256": sha256_file(split_path),
        "checks": checks, "timing": timing,
        **{k: prov[k] for k in ("dataset", "condition", "input_modality", "protocol_id", "manifest_sha256",
                                "manifest", "manifest_file_sha256", "reference_checkpoint_dir",
                                "saved_run_config_sha256", "saved_split_sha256", "split_metadata_sha256")},
        "gold_label_protection": {"embedded_row_label_in_input": False,
                                  "shot_labels_in_input": args.mode == "manyshot",
                                  "shot_subjects": "fit subjects only, never the embedded subject"},
        "versions": {"python": platform.python_version(), "transformers": transformers.__version__},
    }
    from src.utils import write_jsonl  # noqa: PLC0415

    for part, directory, file_name in layout:
        rows, vecs, ans = [], [], []
        for s in plan[part]:
            for e in sorted((pool_rows[k] for k in pool_rows if pool_rows[k]["subject_id"] == s),
                            key=lambda r: r["sample_id"]):
                rows.append({**e, "partition": file_name, "fold": fold, "pooling": POOLING})
                vecs.append(vectors[e["sample_id"]])
                if answer:
                    sc = answer[e["sample_id"]]
                    ans.append({"sample_id": e["sample_id"], "subject_id": e["subject_id"], "label": e["label"],
                                **{f"logprob_{LABEL_TEXT[k]}": float(sc[n]) for n, k in enumerate(ANSWER_ORDER)}})
        parts[part] = write_cache(directory, file_name, rows, vecs)
        if answer:
            write_jsonl(ans, directory / f"{file_name}_answer_scores.jsonl")
    save_json({**meta_common, "partitions": {k: v for k, v in parts.items() if k != "dev"}},
              partial / "extraction_metadata.json")
    if "dev" in plan:
        save_json({**meta_common, "partitions": {"final_eval": parts["dev"]}, "role": "daic_official_dev"},
                  partial / "dev" / "extraction_metadata.json")
    partial.rename(out_dir)
    print("[done]", route["name"], "fold", fold, json.dumps(parts), flush=True)


if __name__ == "__main__":
    main()
