from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0,str(ROOT))
from src.daic_statistics import exact_mcnemar, holm_adjust, stratified_paired_bootstrap
from src.utils import save_json

PAIRS={
    "joint_audio":"joint_all_audio,joint_matched10_audio",
    "joint_audio_text":"joint_all_audio_text,joint_matched10_audio_text",
    "independent_audio_normalized":"independent_all_audio_normalized,independent_matched10_audio_normalized",
    "independent_audio_equal_row":"independent_all_audio_equal_row,independent_matched10_audio_equal_row",
    "independent_audio_text_normalized":"independent_all_audio_text_normalized,independent_matched10_audio_text_normalized",
    "independent_audio_text_equal_row":"independent_all_audio_text_equal_row,independent_matched10_audio_text_equal_row",
}


def read_predictions(path: Path, seed: int) -> list[dict]:
    with path.open(newline="",encoding="utf-8") as handle:
        rows=list(csv.DictReader(handle))
    result=[]
    for row in rows:
        prediction=row.get("prediction","")
        label=int(row["label"])
        # Binary-strict evaluation counts INVALID as wrong. Mapping an invalid
        # prediction to the opposite label preserves that behavior in resamples.
        strict_prediction=int(prediction) if prediction in {"0","1"} else 1-label
        result.append({"subject_id":str(row["subject_id"]),"label":label,
                       "prediction":strict_prediction,"seed":seed})
    return result


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--matrix",type=Path,required=True)
    parser.add_argument("--output-dir",type=Path)
    parser.add_argument("--groups",default="joint,independent")
    args=parser.parse_args()
    matrix=json.loads(args.matrix.read_text())
    selected_groups={value.strip() for value in args.groups.split(",") if value.strip()}
    roots={task["cell_id"]:ROOT/task["output_root"] for task in matrix["tasks"] if task["kind"]=="evaluation" and task.get("group") in selected_groups}
    metrics={}; predictions={}
    for cell,root in roots.items():
        metrics[cell]=json.loads((root/"evaluation/metrics_original_teacher_forced.json").read_text())
        predictions[cell]=read_predictions(root/"evaluation/predictions_subject_level.csv",int(matrix["seed"]))
    comparisons=[]
    for name,pair in PAIRS.items():
        unmatched,matched=pair.split(",")
        if unmatched not in metrics or matched not in metrics: continue
        mcnemar=exact_mcnemar(predictions[unmatched],predictions[matched])
        comparisons.append({"comparison":name,"unmatched":unmatched,"matched":matched,
            "strict_positive_f1_delta":metrics[matched]["binary_strict_positive_f1"]-metrics[unmatched]["binary_strict_positive_f1"],
            "strict_macro_f1_delta":metrics[matched]["binary_strict_macro_f1"]-metrics[unmatched]["binary_strict_macro_f1"],
            "positive_f1_bootstrap":stratified_paired_bootstrap(predictions[unmatched],predictions[matched],metric="positive_f1",iterations=10000,seed=1337),
            "macro_f1_bootstrap":stratified_paired_bootstrap(predictions[unmatched],predictions[matched],metric="macro_f1",iterations=10000,seed=1337),
            "mcnemar":mcnemar})
    adjusted=holm_adjust([row["mcnemar"]["p_value"] for row in comparisons]) if comparisons else []
    for row,value in zip(comparisons,adjusted): row["mcnemar"]["holm_p_value"]=value
    payload={"schema_version":"daic_non_k_report.v1","run_id":matrix["run_id"],"seed":matrix["seed"],
             "metrics":metrics,"comparisons":comparisons,
             "count_only_diagnostic":{"unmatched_rule":"chunks > 10 predicts depressed","unmatched_accuracy":1.0,
                                      "matched_unique_counts":[10],"matched_majority_baseline_accuracy":33/47},
             "interpretation_caveats":[
                 "Matched10 removes five depressed audio segments, so performance deltas combine count removal and reduced audio evidence.",
                 "Class-associated segment versus random_segment preprocessing remains after count matching.",
                 "Joint prompts use neutral count wording; independent examples never expose total subject count in one forward pass.",
             ]}
    output=args.output_dir or args.matrix.parent/"report"
    output.mkdir(parents=True,exist_ok=True); save_json(payload,output/"report.json")
    with (output/"summary.csv").open("w",newline="",encoding="utf-8") as handle:
        fields=["cell","accuracy","positive_f1","macro_f1","precision","recall","invalid_subjects"]
        writer=csv.DictWriter(handle,fieldnames=fields); writer.writeheader()
        for cell,row in sorted(metrics.items()):
            writer.writerow({"cell":cell,"accuracy":row["binary_strict_accuracy"],"positive_f1":row["binary_strict_positive_f1"],
                             "macro_f1":row["binary_strict_macro_f1"],"precision":row["binary_strict_precision"],
                             "recall":row["binary_strict_recall"],"invalid_subjects":row.get("invalid_subjects",0)})
    lines=["# DAIC non-K and matched-10 results","",f"Run: `{matrix['run_id']}`","",
           "| Cell | Accuracy | Positive-F1 | Macro-F1 | Invalid |","|---|---:|---:|---:|---:|"]
    for cell,row in sorted(metrics.items()):
        lines.append(f"| {cell} | {row['binary_strict_accuracy']:.3f} | {row['binary_strict_positive_f1']:.3f} | {row['binary_strict_macro_f1']:.3f} | {row.get('invalid_subjects',0)} |")
    lines.extend(["","Matched-minus-unmatched paired statistics are recorded in `report.json`.","",
                  "Interpretation: matched10 also removes five depressed audio segments, and preprocessing-kind confounding remains."])
    (output/"report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(json.dumps({"output":str(output),"cells":len(metrics),"comparisons":len(comparisons)},sort_keys=True))


if __name__=="__main__": main()
