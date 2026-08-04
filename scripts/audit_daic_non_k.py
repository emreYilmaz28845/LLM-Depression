from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))

from src.daic_chunking import build_independent_epoch_schedule
from src.data.runtime import AUDIO_PLACEHOLDER, build_examples
from src.utils import load_yaml_with_overrides, read_json, save_json
from scripts.build_daic_non_k_matrix import canonical_hash


def args_for(overrides: dict[str, Any]) -> list[str]:
    def scalar(value: Any) -> str:
        if isinstance(value, bool): return "true" if value else "false"
        if isinstance(value, (list, dict)): return json.dumps(value, separators=(",", ":"))
        return str(value)
    result=[]
    for key,value in overrides.items(): result.extend(["--set", f"{key}={scalar(value)}"])
    return result


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--groups", default="joint,independent")
    parser.add_argument("--require-artifacts", action="store_true")
    parser.add_argument("--slurm-accounting", type=Path)
    parser.add_argument("--output", type=Path)
    args=parser.parse_args()
    matrix=read_json(args.matrix)
    failures=[]; checks={}
    matrix_for_hash=dict(matrix); recorded_hash=matrix_for_hash.pop("matrix_hash",None)
    if not recorded_hash or canonical_hash(matrix_for_hash)!=recorded_hash:
        failures.append("matrix_hash_mismatch")
    selected_groups={value.strip() for value in args.groups.split(",") if value.strip()}
    if not selected_groups or not selected_groups <= {"joint","independent"}: failures.append("invalid_groups")
    train_tasks=[task for task in matrix.get("tasks",[]) if task.get("kind")=="train" and task.get("group") in selected_groups]
    eval_tasks=[task for task in matrix.get("tasks",[]) if task.get("kind")=="evaluation" and task.get("group") in selected_groups]
    expected=sum(task.get("kind")=="train" and task.get("group") in selected_groups for task in matrix.get("tasks",[]))
    if len(train_tasks)!=expected or len(eval_tasks)!=expected: failures.append("wrong_task_count")
    if len({task["cell_id"] for task in train_tasks})!=expected: failures.append("duplicate_cell_id")
    manifest=args.manifest or args.matrix.parent/"shared/manifests/daic_manifest.jsonl"
    if manifest.is_file():
        rows=[json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
        grouped=defaultdict(list)
        for row in rows: grouped[str(row["subject_id"])].append(row)
        dist=Counter((int(items[0]["label"]),len(items)) for items in grouped.values())
        if dist != Counter({(0,10):133,(1,15):56}): failures.append(f"manifest_distribution:{dict(dist)}")
        cell_checks={}
        for task in train_tasks:
            config=load_yaml_with_overrides(ROOT/task["base_config"], args_for(task["overrides"]))
            train_rows=[row for row in rows if row["split_original"]=="train"]
            examples=build_examples(train_rows, config, "train")
            matched="matched10" in task["protocol_id"]
            if config["data"]["sample_mode"]=="subject_audio":
                counts=Counter(len(example["audio_paths"]) for example in examples)
                expected_counts=Counter({10:107}) if matched else Counter({10:77,15:30})
                if counts!=expected_counts: failures.append(f"{task['cell_id']}:joint_counts:{dict(counts)}")
                for example in examples:
                    if example["prompt_text"].count(AUDIO_PLACEHOLDER)!=len(example["audio_paths"]):
                        failures.append(f"{task['cell_id']}:placeholder_mismatch:{example['subject_id']}")
                    if "provided in 10 segments" in example["prompt_text"] or "provided in 15 segments" in example["prompt_text"]:
                        failures.append(f"{task['cell_id']}:explicit_count_text:{example['subject_id']}")
                cell_checks[task["cell_id"]]={"examples":len(examples),"audio_counts":dict(counts)}
            else:
                controls=config["data"]
                schedules,audit=build_independent_epoch_schedule(
                    examples, policy=controls["train_chunk_policy"],
                    chunks_per_subject=controls["train_chunks_per_subject"], seed=1337, epochs=1,
                    loss_weight_rescale=controls["loss_weight_rescale"],
                    equal_row_weight=bool(controls["equal_row_weight"]),
                )
                per_subject=Counter(row["subject_id"] for row in schedules[0])
                count_dist=Counter(per_subject.values())
                expected_counts=Counter({10:107}) if matched else Counter({10:77,15:30})
                if count_dist!=expected_counts: failures.append(f"{task['cell_id']}:independent_counts:{dict(count_dist)}")
                if bool(controls["equal_row_weight"]) == bool(audit["equal_total_subject_weight"]):
                    failures.append(f"{task['cell_id']}:weight_policy_mismatch")
                cell_checks[task["cell_id"]]={"examples":len(schedules[0]),"subject_count_distribution":dict(count_dist)}
        checks["cells"]=cell_checks
    elif args.require_artifacts:
        failures.append("missing_manifest")
    if args.require_artifacts:
        for task in train_tasks:
            root=ROOT/task["output_root"]
            for path in (root/"best_model",root/"run_config.yaml",root/"logs/split_used.json"):
                if not path.exists(): failures.append(f"missing:{path}")
        for task in eval_tasks:
            root=ROOT/task["output_root"]/"evaluation"
            for path in (root/"metrics_original_teacher_forced.json",root/"predictions_subject_level.csv"):
                if not path.is_file(): failures.append(f"missing:{path}")
    if args.slurm_accounting:
        accounting=[json.loads(line) for line in args.slurm_accounting.read_text().splitlines() if line.strip()]
        by_task={row["task_id"]:row for row in accounting}
        for task in matrix["tasks"]:
            if task.get("group") not in selected_groups: continue
            row=by_task.get(task["task_id"])
            if not row: failures.append(f"missing_slurm:{task['task_id']}")
            elif row.get("state")!="COMPLETED" or row.get("exit_code")!="0:0": failures.append(f"bad_slurm:{task['task_id']}:{row.get('state')}")
    payload={"schema_version":"daic_non_k_audit.v1","run_id":matrix["run_id"],"stage":matrix["stage"],"passed":not failures,"failures":failures,"checks":checks}
    output=args.output or args.matrix.parent/f"audit_{matrix['stage']}.json"
    save_json(payload,output); print(json.dumps(payload,indent=2,sort_keys=True)); raise SystemExit(0 if payload["passed"] else 1)


if __name__=="__main__": main()
