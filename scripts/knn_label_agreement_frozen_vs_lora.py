"""Held-out K-nearest-neighbour label agreement, frozen vs LoRA text embeddings.

For every fold, subject-mean embeddings of ``outer_train`` and ``final_eval`` are
z-normalized with the train statistics. Each held-out subject is a query and its K
nearest train subjects (cosine) are the neighbours. The score is the share of
neighbours with the query's label. Descriptive diagnostic, not classifier accuracy.

Differs from the within-corpus all-subject kNN in the text_kaan_replication E3 report:
queries are held-out subjects and neighbours are train subjects of the same fold, so
the LoRA model never saw the queries.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

CORPORA = {
    "cmdc": "Mandarin (CMDC)",
    "androids_interview": "Italian (Androids)",
    "daic": "English (DAIC-WOZ)",
    "turkish": "Turkish",
    "d3tec": "Spanish (D3TEC)",
}


def load_subject_means(fold_dir: Path, part: str):
    vectors = np.load(fold_dir / f"{part}.npz", allow_pickle=True)["vectors"]
    rows = [json.loads(line) for line in (fold_dir / f"{part}_rows.jsonl").read_text().splitlines()]
    by_subject: dict = {}
    for vec, row in zip(vectors, rows):
        by_subject.setdefault((row["subject_id"], row["label"]), []).append(vec)
    keys = list(by_subject)
    x = np.stack([np.mean(by_subject[k], axis=0) for k in keys])
    y = np.array([k[1] for k in keys])
    return x, y


def agreement(root: Path, corpus: str, view: str, k: int):
    base = root / corpus / view
    if not base.is_dir():
        return None
    agree, chance = [], []
    for fold_dir in sorted(p for p in base.iterdir() if p.is_dir()):
        try:
            xt, yt = load_subject_means(fold_dir, "outer_train")
            xe, ye = load_subject_means(fold_dir, "final_eval")
        except FileNotFoundError:
            continue
        mu, sd = xt.mean(0), xt.std(0) + 1e-6
        xt, xe = (xt - mu) / sd, (xe - mu) / sd
        xt /= np.linalg.norm(xt, axis=1, keepdims=True)
        xe /= np.linalg.norm(xe, axis=1, keepdims=True)
        nn = np.argsort(-(xe @ xt.T), axis=1)[:, : min(k, len(yt))]
        agree += list((yt[nn] == ye[:, None]).mean(1))
        p = yt.mean()
        chance += list(np.where(ye == 1, p, 1 - p))
    if not agree:
        return None
    return float(np.mean(agree)), float(np.mean(chance)), len(agree)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frozen-root", type=Path, required=True, help="frozen_fold_caches")
    ap.add_argument("--lora-root", type=Path, required=True, help="LoRA caches (use the corrected text rerun)")
    ap.add_argument("--k", type=int, default=10)
    args = ap.parse_args()
    for view in ("text_only_native", "text_only_english"):
        print(f"\n{view}\n| Corpus | Frozen | LoRA | Chance | Queries |\n|---|---|---|---|---|")
        for corpus, name in CORPORA.items():
            frozen = agreement(args.frozen_root, corpus, view, args.k)
            lora = agreement(args.lora_root, corpus, view, args.k)
            if frozen is None or lora is None:
                print(f"| {name} | not available | not available | | |")
                continue
            print(f"| {name} | {frozen[0]:.3f} | {lora[0]:.3f} | {frozen[1]:.3f} | {frozen[2]} |")


if __name__ == "__main__":
    main()
