"""Build per-cell outer/inner predictions from LoRA-embedding logistic heads.

Same structure as margin_selector_analysis.load_cells, so the selector and ensemble
analyses can run unchanged on a second decision rule: an L2 logistic head on subject-mean
LoRA-adapter embeddings instead of the LLM's candidate-label likelihood. Margin = head logit.

Sources (native language): seed 1337 text from heads_lora_textrerun, seed 1337 audio
modalities from heads_lora, seeds 7 and 2024 from heads_lora_seeds/<ds>/<mod>/s<seed>.

Inner (selection) predictions, by corpus protocol:
- train_val_test (Androids, D3TEC): head logits on the split's selection subjects, which are
  the LLM's inner-validation subjects (not LoRA-trained; the head's L2 was chosen on them).
- train_val (Turkish, CMDC): inner = outer, as for the LLM; downstream code then uses
  leave-one-fold-out.
- train_cv_test (DAIC): the LoRA cache has no dev-subject embeddings, so inner = 5-fold
  inner-CV out-of-fold logits on the training subjects. These subjects were LoRA-trained,
  so these predictions are optimistic; DAIC selector results are flagged as not comparable.

Every head is refit from the cache with its saved L2 penalty and must reproduce the saved
outer logits.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from baselines import frozen_standalone_heads as fsh  # noqa: E402

SEEDS = (7, 1337, 2024)
CORPORA = ("cmdc", "androids_interview", "daic", "turkish", "d3tec")
TOL = 1e-6


def route_dir(root, ds, mod, seed):
    if seed == 1337:
        base = "heads_lora_textrerun" if mod == "text_only" else "heads_lora"
        return root / base / ds / f"{mod}_native"
    return root / "heads_lora_seeds" / ds / f"{mod}_native" / f"s{seed}"


def _path(p, source):
    p = Path(p)
    return p if p.is_absolute() else source / p


def seed_of_checkpoint(ckpt):
    m = re.search(r"_s(\d+)_f\d", str(ckpt))
    return int(m.group(1)) if m else None


def fold_cell(source, rdir, cfg, fold, record):
    cd = _path(cfg["cache_identities"][fold]["outer_train.npz"]["path"], source).parent
    cache, meta = fsh._load_caches(cd, None, True)
    record(cd / "extraction_metadata.json", "lora_extraction_metadata")
    split_path = _path(cfg["split_files"][fold]["path"], source)
    split = json.loads(split_path.read_text()); record(split_path, "head_split")
    if fsh.canonical_sha256(split) != cfg["split_files"][fold]["sha256"]:
        raise ValueError("Split hash mismatch: " + str(split_path))
    sel = json.loads((rdir / f"fold_{fold}/selection.json").read_text())
    record(rdir / f"fold_{fold}/selection.json", "head_selection")
    penalty, protocol = sel["selected_l2_penalty"], cfg["protocol"]
    train = sorted(map(str, split["train_subject_ids"]))
    selection = sorted(map(str, split.get("selection_subject_ids", [])))
    report = selection if protocol == "train_val" else sorted(map(str, split["final_eval_subject_ids"]))

    head = fsh._fit(cache, train, penalty)
    saved_path = rdir / f"fold_{fold}/predictions_subject_level.csv"; record(saved_path, "head_outer_predictions")
    saved = pd.read_csv(saved_path, dtype={"subject_id": str}).set_index("subject_id")
    if set(saved.index) != set(report):
        raise ValueError("Saved head subjects differ from split report subjects: " + str(saved_path))
    logit = fsh._logits(cache, head, report)
    if np.max(np.abs(logit - saved.loc[report, "mean_logit"].to_numpy())) > TOL:
        raise ValueError("Refit head does not reproduce saved logits: " + str(saved_path))
    outer = pd.DataFrame({"subject_id": report, "label": cache.y(report), "margin": logit})

    if protocol == "train_val":
        inner, inner_mode = outer.copy(), "outer_equals_inner"
    elif protocol == "train_val_test":
        cache.require(selection, "selection subjects")
        inner = pd.DataFrame({"subject_id": selection, "label": cache.y(selection),
                              "margin": fsh._logits(cache, head, selection)})
        inner_mode = "selection_subjects"
    elif protocol == "train_cv_test":
        folds = sel.get("inner_folds") or fsh.inner_cv_folds(train, cache.y(train))
        if sorted(s for f in folds for s in f) != train:
            raise ValueError("Inner CV folds do not partition training subjects")
        parts = []
        for held in folds:
            fit = [s for s in train if s not in set(held)]
            h = fsh._fit(cache, fit, penalty)
            parts.append(pd.DataFrame({"subject_id": held, "label": cache.y(held), "margin": fsh._logits(cache, h, held)}))
        inner, inner_mode = pd.concat(parts, ignore_index=True), "inner_cv_oof_on_lora_trained_subjects"
    else:
        raise ValueError("Unexpected protocol " + protocol)
    for d in (outer, inner):
        d["prediction"] = (d.margin > 0).astype(int)
    return {"outer": outer[["subject_id", "label", "prediction", "margin"]],
            "inner": inner[["subject_id", "label", "prediction", "margin"]],
            "inner_mode": inner_mode, "protocol": protocol, "l2_penalty": penalty,
            "checkpoint": str(meta.get("checkpoint_dir")), "cache": str(cd)}


def load_lora_cells(source, record, mods):
    root = source / "output_model/frozen_base_standalone_heads"
    out = {}
    for ds in CORPORA:
        for mod in mods:
            for seed in SEEDS:
                rdir = route_dir(root, ds, mod, seed)
                cfg = json.loads((rdir / "result_config.json").read_text()); record(rdir / "result_config.json", "head_config")
                if cfg.get("feature_source") != "lora_adapter":
                    raise ValueError("Not a LoRA head: " + str(rdir))
                for fold in range(len(cfg["split_files"])):
                    c = fold_cell(source, rdir, cfg, fold, record)
                    s = seed_of_checkpoint(c["checkpoint"])
                    if s is not None and s != seed:
                        raise ValueError(f"Checkpoint seed {s} != {seed} in {rdir} fold {fold}")
                    c.update(cell_id=f"L-{ds}-{mod}-{seed}-{fold}", fold_dir=str(rdir), selected_epoch=None,
                             inner_file=c["inner_mode"], checkpoint_seed=s)
                    out[(ds, seed, fold, mod)] = c
                print("lora heads", ds, mod, seed, flush=True)
    return out


def frozen_identity_audit(source, record):
    """Text LoRA subject means must differ from frozen text subject means (old cache bug)."""
    root = source / "output_model/frozen_base_standalone_heads"
    rows = []
    for ds in CORPORA:
        fcfg = json.loads((root / "heads" / ds / "text_only_native/result_config.json").read_text())
        ci = fcfg.get("cache_identity") or fcfg["cache_identities"][0]
        fcd = _path(ci["outer_train.npz"]["path"], source).parent
        di = fcfg.get("dev_cache_identity")
        dev = _path(di["final_eval.npz"]["path"], source).parent if di else None
        frozen, _ = fsh._load_caches(fcd, dev, False)
        for seed in SEEDS:
            rdir = route_dir(root, ds, "text_only", seed)
            cfg = json.loads((rdir / "result_config.json").read_text())
            for fold in range(len(cfg["split_files"])):
                cd = _path(cfg["cache_identities"][fold]["outer_train.npz"]["path"], source).parent
                lora, _ = fsh._load_caches(cd, None, True)
                subs = sorted(set(lora.units) & set(frozen.units))
                a = lora.subject_means(lora.x, subs); b = frozen.subject_means(frozen.x, subs)
                cos = np.sum(a * b, 1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
                rel = np.linalg.norm(a - b, axis=1) / np.linalg.norm(b, axis=1)
                rows.append({"corpus": ds, "seed": seed, "fold": fold, "n_subjects": len(subs),
                             "median_cosine_lora_vs_frozen": float(np.median(cos)),
                             "median_relative_l2_difference": float(np.median(rel)),
                             "identical_to_frozen": bool(np.median(rel) < 1e-4)})
    return pd.DataFrame(rows)
