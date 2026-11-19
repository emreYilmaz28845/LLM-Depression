"""Metrics of the model's own answer (candidate-label likelihood) from `<partition>_answer_scores.jsonl`.

Prediction = the label with the higher mean token log-probability (the repo's likelihood rule); rows of
one subject are averaged before the decision. Usage: python answer_metrics.py CACHE_DIR [CACHE_DIR ...]
Reads final_eval (and dev/final_eval when present) of each cache dir and prints one JSON line per cache.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

from sklearn.metrics import f1_score, recall_score


def subject_scores(path: Path):
    acc = defaultdict(lambda: [0.0, 0.0, 0, None])
    for line in path.read_text().splitlines():
        r = json.loads(line)
        a = acc[r["subject_id"]]
        a[0] += r["logprob_Depressed"]; a[1] += r["logprob_Non-depressed"]; a[2] += 1
        if a[3] is not None and a[3] != r["label"]:
            raise ValueError(f"Conflicting labels for a subject in {path}")
        a[3] = r["label"]
    y = [a[3] for a in acc.values()]
    pred = [int(a[0] / a[2] > a[1] / a[2]) for a in acc.values()]
    return y, pred


def metrics(y, pred):
    return {"n_subjects": len(y), "macro_f1": round(f1_score(y, pred, average="macro"), 4),
            "positive_f1": round(f1_score(y, pred, pos_label=1, zero_division=0), 4),
            "uar": round(recall_score(y, pred, average="macro"), 4),
            "predicted_positive": int(sum(pred)), "true_positive_subjects": int(sum(y))}


if __name__ == "__main__":
    for d in map(Path, sys.argv[1:]):
        out = {"cache": str(d), "test": metrics(*subject_scores(d / "final_eval_answer_scores.jsonl"))}
        dev = d / "dev" / "final_eval_answer_scores.jsonl"
        if dev.exists():
            out["dev"] = metrics(*subject_scores(dev))
        print(json.dumps(out))
