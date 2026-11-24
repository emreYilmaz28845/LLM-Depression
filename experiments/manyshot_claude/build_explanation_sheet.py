"""Explanation workbook for one model's DAIC runs (advisor asked for one explanations sheet per model).

One tab per run (e.g. many-shot and zero-shot with the advisor's prompt) plus a Setup tab with metrics.
Rows are test subjects, errors first, with the gold label, PHQ-8 score, model answer and the explanation text
unchanged. Many-shot tabs also list the example numbers the explanation cites, with each example's real label
and PHQ-8 score (checked against the shot order of the run). Subject IDs are left out.

Usage:
  python experiments/manyshot_claude/build_explanation_sheet.py --out OUT.xlsx \
      --run "Many-shot (advisor prompt)=outputs/daic_claude_advisor_manyshot_effmedium" \
      --run "Zero-shot (advisor prompt)=outputs/daic_claude_advisor_zeroshot_effmedium" \
      --phq-csv TRAIN.csv --phq-csv DEV.csv --phq-csv TEST.csv
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import random
import re
from pathlib import Path

_spec = importlib.util.spec_from_file_location("runner", Path(__file__).with_name("run_claude_manyshot.py"))
runner = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(runner)

CASE_ORDER = {"FN": 0, "FP": 1, "TP": 2, "TN": 3}


def load_phq(paths: list[Path]) -> dict[str, int]:
    phq: dict[str, int] = {}
    for path in paths:
        with path.open(newline="") as f:
            for row in csv.DictReader(f):
                phq[str(int(row["Participant_ID"]))] = int(row["PHQ8_Score"])
    return phq


def shot_order(run_info: dict) -> list[dict]:
    """Rebuild the run's numbered shot list exactly as run_claude_manyshot.py built it."""
    subjects = runner.load_subjects(Path(run_info["manifest"]))
    shots = sorted((s for s in subjects.values() if s["split"] in runner.SHOT_SPLITS), key=lambda s: s["subject_id"])
    random.Random(run_info["shot_seed"]).shuffle(shots)
    if runner.sha256_text(runner.shot_block(shots)) != run_info["shot_block_sha256"]:
        raise ValueError("Rebuilt shot block does not match the run's shot_block_sha256.")
    return shots


def cited_examples(text: str, n_shots: int) -> list[int]:
    """Example numbers an explanation cites: 'Example 12', 'Examples 7, 14 and 43', 'examples (e.g., 47, 44)'.

    Reads the numbers after each 'example(s)' up to the end of that clause ('.', ';', ':' or ')'), ignoring PHQ-8
    values, and keeps numbers between 1 and n_shots.
    """
    t = text.replace("e.g.", "eg").replace("i.e.", "ie")
    out: set[int] = set()
    for m in re.finditer(r"examples?\b", t, flags=re.I):
        clause = re.split(r"[.;:)]", t[m.end():])[0][:200]
        clause = re.sub(r"PHQ-?8[^\d]{0,30}\d*", "", clause)
        out.update(int(n) for n in re.findall(r"(?<![\w-])(\d{1,3})(?!\d)", clause) if 1 <= int(n) <= n_shots)
    return sorted(out)


def case(gold: int, pred: int | None) -> str:
    if pred is None:
        return "INVALID"
    return {(1, 1): "TP", (1, 0): "FN", (0, 1): "FP", (0, 0): "TN"}[(gold, pred)]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--run", action="append", required=True, help="'Tab title=run_dir'")
    p.add_argument("--phq-csv", action="append", required=True, type=Path)
    args = p.parse_args()

    from openpyxl import Workbook  # noqa: PLC0415
    from openpyxl.styles import Alignment, Font  # noqa: PLC0415

    phq = load_phq(args.phq_csv)
    wb = Workbook()
    setup = wb.active
    setup.title = "Setup"
    setup.append(["Run", "Model", "Prompt", "Shots", "Test subjects", "macro-F1", "positive-F1", "UAR",
                  "TP", "FN", "FP", "TN", "INVALID"])
    for spec in args.run:
        title, run_dir = spec.split("=", 1)
        run_dir = Path(run_dir)
        info = json.loads((run_dir / "run_info.json").read_text())
        met = json.loads((run_dir / "metrics.json").read_text())
        if not met.get("complete"):
            raise ValueError(f"{run_dir}: run is not complete.")
        recs = [json.loads(line) for line in (run_dir / "responses.jsonl").read_text().splitlines()]
        shots = shot_order(info) if info["n_shots"] else []
        counts = {k: 0 for k in ("TP", "FN", "FP", "TN", "INVALID")}
        rows = []
        for r in recs:
            c = case(r["label"], r["pred"])
            counts[c] += 1
            row = [c, runner.LABEL_TEXT[r["label"]], phq[r["subject_id"]], r["pred_text"]]
            if shots:
                nums = cited_examples(r["explanation"], len(shots))
                row.append("; ".join(f"Example {n} ({runner.LABEL_TEXT[shots[n - 1]['label']]}, "
                                     f"PHQ-8 {phq[shots[n - 1]['subject_id']]})" for n in nums) or "none")
            row.append(r["explanation"])
            rows.append(row)
        rows.sort(key=lambda row: (CASE_ORDER.get(row[0], -1), -row[2]))
        setup.append([title, info["model"] + f" (effort {info['effort']})", info["prompt_version"], info["n_shots"],
                      met["n_subjects"], met["macro_f1"], met["positive_f1"], met["uar"],
                      counts["TP"], counts["FN"], counts["FP"], counts["TN"], counts["INVALID"]])
        ws = wb.create_sheet(title[:31])
        header = ["Case", "True label", "PHQ-8 score", "Model answer"]
        header += (["Examples cited (real label, PHQ-8)"] if shots else []) + ["Explanation"]
        ws.append(header)
        for row in rows:
            ws.append(row)
        for cell in ws[1]:
            cell.font = Font(bold=True)
        widths = [9, 15, 12, 15] + ([45] if shots else []) + [110]
        for i, w in enumerate(widths):
            ws.column_dimensions[chr(65 + i)].width = w
        for row in ws.iter_rows(min_row=2):
            for cell in row:
                cell.alignment = Alignment(wrap_text=True, vertical="top")
        ws.freeze_panes = "A2"
    for cell in setup[1]:
        cell.font = Font(bold=True)
    setup.append([])
    setup.append(["DAIC-WOZ official test (47 subjects). Depressed = PHQ-8 >= 10. Shots = all 142 official "
                  "train+dev subjects, numbered. Advisor prompt sentence: '" + runner.ADVISOR_SENTENCE + "'. "
                  "Rows are sorted errors first (FN, FP, TP, TN). Explanation text is the model's, unchanged."])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    wb.save(args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
