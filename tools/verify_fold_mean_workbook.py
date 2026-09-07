"""Verify Qwen TF workbook constants from an explicit, hashed fold evidence index.

No model runs, source artifact edits, or registry status changes. The index
must identify exactly five distinct folds for each native/English IT/ES cell.
"""
import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import build_clean_workbook as wb


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = json.loads(args.evidence_index.read_text())["records"]
    selected = []
    ds_names = {"d3tec": "D3TEC", "androids_interview": "Androids Interview"}
    mod_names = {"audio_text": "Audio + Text", "audio_only": "Audio only", "text_only": "Text only"}
    for row in records:
        if row["dataset"] not in ds_names or row["model"] != "qwen" or row["route"] != "teacher_forced":
            continue
        values, folds = [], []
        for ref in row["fold_evidence"]:
            path = Path(ref["metrics_path"])
            assert hashlib.sha256(path.read_bytes()).hexdigest() == ref["metrics_sha256"], path
            cfg = Path(ref["config_path"])
            assert hashlib.sha256(cfg.read_bytes()).hexdigest() == ref["config_sha256"], cfg
            folds.append(int(cfg.parent.name.removeprefix("fold_")))
            data = json.loads(path.read_text())
            cm = data["binary_strict_confusion_matrix"]
            tn, fp = cm[0][:2]
            fn, tp = cm[1][:2]
            neg_invalid = cm[0][2] if len(cm[0]) > 2 else 0
            pos_invalid = cm[1][2] if len(cm[1]) > 2 else 0
            pos = 2 * tp / (2 * tp + fp + fn + pos_invalid) if 2 * tp + fp + fn + pos_invalid else 0
            neg = 2 * tn / (2 * tn + fp + fn + neg_invalid) if 2 * tn + fp + fn + neg_invalid else 0
            computed = [(pos + neg) / 2, pos]
            assert all(abs(a - data["binary_strict_" + key]) < 1e-9 for a, key in zip(computed, ["macro_f1", "positive_f1"]))
            values.append(computed)
        assert sorted(folds) == list(range(5)), folds
        mean = [statistics.mean(v[i] for v in values) for i in range(2)]
        key = ds_names[row["dataset"]], mod_names[row["modality"]]
        expected = [wb.STANDALONE_QWEN[key], wb.STANDALONE_QWEN_POSF1[key]] if row["condition"] == "native" else [wb.EN_TF[key][0], wb.EN_TF_POSF1[key][0]]
        assert all(abs(a - b) < 1e-12 for a, b in zip(mean, expected)), key
        selected.append({**row, "previous_display": row["display"], "macro_f1": mean[0], "positive_f1": mean[1], "aggregation": "unweighted five-fold mean", "fold_values": values})
    assert len(selected) == 10
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"status": "passed", "source_index": str(args.evidence_index), "source_index_sha256": hashlib.sha256(args.evidence_index.read_bytes()).hexdigest(), "records": selected}, indent=2) + "\n")
    print("Verified 10 paired constants from 50 hash-checked fold/config artifacts")


if __name__ == "__main__":
    main()
