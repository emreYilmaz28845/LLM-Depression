#!/usr/bin/env python3
"""Production submission driver for the native-versus-English head study.

Phases (locked matrix = 1248 jobs):
  chains : standalone train+eval chains (12 panels x 4 ds x 5 folds = 240
           chains -> 480 jobs) and merged CV chains (4 variants x 3 seeds
           x 5 folds = 60 chains -> 180 jobs)
  heads  : standalone logreg (240) + standalone optuna-100 (240) with
           --after-job-id chaining from phase `chains` records
  final  : merged final chains + optuna (48) — requires completed CV

State file records every submitted identity so resumption skips completed
submissions (submission is idempotent per run/fold/attempt contract).
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import tools.native_en_submit as ns

SLUG_DEFAULT = "exp-native-en-text-heads-20260822"
SCHEDULER_DEFAULT = "ozu647717@alogin2.bsc.es"
STANDALONE_DATASETS = ("d3tec", "androids_interview", "cmdc", "turkish")
FOLDS = (0, 1, 2, 3, 4)
SEEDS = (7, 1337, 2024)
VARIANTS = ("native_qwen", "english_qwen", "native_gemma4", "english_gemma4")

PANEL_CONFIG = {
    ("native", "qwen"): {
        "d3tec": "configs/main/d3tec_text_only_harmonized_selmacrof1_tf.yaml",
        "androids_interview": "configs/main/androids_text_only_harmonized_selmacrof1_tf.yaml",
        "cmdc": "configs/main/cmdc_text_only_harmonized_selmacrof1_tf.yaml",
        "turkish": "configs/main/turkish_t17_text_only_harmonized_selmacrof1_tf_qwen3asr.yaml",
    },
    ("english", "qwen"): {
        "d3tec": "configs/main/d3tec_text_only_harmonized_selmacrof1_tf_en.yaml",
        "androids_interview": "configs/main/androids_text_only_harmonized_selmacrof1_tf_en.yaml",
        "cmdc": "configs/main/cmdc_text_only_harmonized_selmacrof1_tf_en.yaml",
        "turkish": "configs/main/turkish_t17_text_only_harmonized_selmacrof1_tf_qwen3asr_en.yaml",
    },
    ("native", "gemma4"): {
        "d3tec": "configs/main/d3tec_text_only_harmonized_selmacrof1_tf_gemma4_12b.yaml",
        "androids_interview": "configs/main/androids_text_only_harmonized_selmacrof1_tf_gemma4_12b.yaml",
        "cmdc": "configs/main/cmdc_text_only_harmonized_selmacrof1_tf_gemma4_12b.yaml",
        "turkish": "configs/main/turkish_t17_text_only_harmonized_selmacrof1_tf_qwen3asr_gemma4_12b.yaml",
    },
    ("english", "gemma4"): {
        "d3tec": "configs/main/d3tec_text_only_harmonized_selmacrof1_tf_en_gemma4_12b.yaml",
        "androids_interview": "configs/main/androids_text_only_harmonized_selmacrof1_tf_en_gemma4_12b.yaml",
        "cmdc": "configs/main/cmdc_text_only_harmonized_selmacrof1_tf_en_gemma4_12b.yaml",
        "turkish": "configs/main/turkish_t17_text_only_harmonized_selmacrof1_tf_qwen3asr_en_gemma4_12b.yaml",
    },
}

LOCKED_JOBS = {"chains": 660, "heads": 540, "final": 48, "total": 1248}


def run(cmd: list[str], log: Path) -> str:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    log.write_text(proc.stdout + ("\n[stderr]\n" + proc.stderr if proc.stderr else ""))
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout).strip().splitlines()[-4:]
        raise RuntimeError(f"{' '.join(cmd[:5])}… failed: " + " | ".join(tail))
    return proc.stdout


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--deployment-id", required=True)
    ap.add_argument("--phase", required=True, choices=("chains", "heads", "final"))
    ap.add_argument("--scheduler-host", default=SCHEDULER_DEFAULT)
    ap.add_argument("--state", default=str(PROJECT_ROOT / "outputs/native_en_production/state.json"))
    ap.add_argument("--limit", type=int, default=0, help="submit at most N cells this invocation")
    args = ap.parse_args()

    state_path = Path(args.state)
    state = json.loads(state_path.read_text()) if state_path.is_file() else {
        "schema_version": "audiollm.native_en_production_state.v1",
        "deployment_id": args.deployment_id,
        "chains": {}, "heads": {}, "final": {},
    }
    state.setdefault(args.phase, {})
    out_dir = state_path.parent / "logs" / args.phase
    out_dir.mkdir(parents=True, exist_ok=True)

    slug = SLUG_DEFAULT
    dep = args.deployment_id
    submitted = 0

    def base() -> list[str]:
        return [sys.executable, str(PROJECT_ROOT / "tools/exp.py")]

    if args.phase == "chains":
        for cond in ("native", "english"):
            for bb in ("qwen", "gemma4"):
                campaign = ns.campaign_for(cond, bb)
                for ds in STANDALONE_DATASETS:
                    cfg = PANEL_CONFIG[(cond, bb)][ds]
                    for fold in FOLDS:
                        for seed in SEEDS:
                            key = f"sa/{cond}-{bb}/{ds}/f{fold}/s{seed}"
                            if key in state["chains"]:
                                continue
                            run_name = f"tnh-{cond[:3] if cond=='native' else 'en'}-{bb}-{ds}-s{seed}"
                            extra = ["--set", "evaluation.evaluation_view=harmonized_all_windows_full_coverage"] if bb == "qwen" else []
                            cmd = base() + [
                                "submit", slug, "--config", cfg, "--fold", str(fold),
                                "--seed", str(seed), "--run-name", run_name,
                                "--campaign", campaign, "--modality", "text_only",
                                "--dataset", ds, "--deployment-id", dep,
                                "--scheduler-host", args.scheduler_host, *extra,
                                "--execute",
                            ]
                            out = run(cmd, out_dir / (key.replace("/", "_") + ".log"))
                            m = re.findall(r"submitted jobs: (\{[^\n]*\})", out)[-1]
                            ids = json.loads(m.replace("'", '"'))
                            state["chains"][key] = {
                                "run_name": run_name, "campaign": campaign,
                                "train_job": ids["train"], "eval_job": ids["best_eval"],
                            }
                            submitted += 2
                            if args.limit and submitted >= args.limit:
                                break
                    # merged chains per variant/seed/fold
        for variant in VARIANTS:
            cond, bb = variant.split("_")
            for seed in SEEDS:
                for fold in FOLDS:
                    key = f"mg/{variant}/f{fold}/s{seed}"
                    if key in state["chains"]:
                        continue
                    run_id = f"tmh-{variant}-s{seed}"
                    cmd = base() + [
                        "submit-merged", slug,
                        "--config", f"configs/experiments/merged/symmetric_merged_text_heads_{variant}.yaml",
                        "--stage", "cv", "--run-id", run_id, "--seed", str(seed),
                        "--condition", cond, "--backbone", bb,
                        "--folds", str(fold), "--deployment-id", dep,
                        "--scheduler-host", args.scheduler_host, "--execute",
                    ]
                    out = run(cmd, out_dir / (key.replace("/", "_") + ".log"))
                    m = re.findall(r"fold (\d+) jobs: (\{[^\n]*\})", out)[-1]
                    ids = json.loads(m[1].replace("'", '"'))
                    state["chains"][key] = {
                        "run_id": run_id, "variant": variant, "seed": seed,
                        "train_job": ids["train"], "post_job": ids["postprocess"],
                        "head_job": ids["head"],
                    }
                    submitted += 3
                    if args.limit and submitted >= args.limit:
                        break

    elif args.phase == "heads":
        chains = state.get("chains") or {}
        for key, rec in sorted(chains.items()):
            if not key.startswith("sa/"):
                continue
            _, panel, ds, fold, seed = key.split("/")
            cond, bb = panel.split("-")
            campaign = ns.campaign_for(cond, bb)
            hkey = f"logreg/{key}"
            okey = f"optuna/{key}"
            run_name = rec["run_name"]
            parent_fold = (
                f"{ns.REMOTE_PROJECT_BASE}/output_model/{campaign}/text_only/{ds}/{run_name}/{fold}"
            )
            if hkey not in state["heads"]:
                cmd = base() + [
                    "submit-hidden", slug,
                    "--parent-fold-dir", parent_fold,
                    "--dataset", ds, "--condition", cond, "--backbone", bb,
                    "--run-name", run_name, "--fold", fold.lstrip("f"),
                    "--seed", seed.lstrip("s"),
                    "--after-job-id", rec["eval_job"],
                    "--deployment-id", dep, "--scheduler-host", args.scheduler_host,
                    "--execute",
                ]
                out = run(cmd, out_dir / (hkey.replace("/", "_") + ".log"))
                jid = re.search(r"submitted logreg attempt job (\d+)", out).group(1)
                state["heads"][hkey] = {"job": jid}
                submitted += 1
            if okey not in state["heads"]:
                cache = (
                    f"{ns.REMOTE_PROJECT_BASE}/outputs/hidden_classifiers/"
                    f"{ds}/{cond}/{run_name}/{fold}/hidden_features"
                )
                cmd = base() + [
                    "submit-optuna100", slug,
                    "--family", "standalone", "--condition", cond, "--backbone", bb,
                    "--dataset", ds, "--run-name", run_name, "--fold", fold.lstrip("f"),
                    "--seed", seed.lstrip("s"), "--target-trials", "100",
                    "--parent-checkpoint-path", f"{parent_fold}/best_model",
                    "--cache-dir", cache,
                    "--after-job-id", state["heads"][hkey]["job"],
                    "--deployment-id", dep, "--scheduler-host", args.scheduler_host,
                    "--execute",
                ]
                out = run(cmd, out_dir / (okey.replace("/", "_") + ".log"))
                jid = re.search(r"submitted optuna attempt job (\d+)", out).group(1)
                state["heads"][okey] = {"job": jid}
                submitted += 1
            if args.limit and submitted >= args.limit:
                break
        # merged Optuna-100 studies (chained to each fold's postprocess)
        for key, rec in sorted(chains.items()):
            if not key.startswith("mg/"):
                continue
            okey = f"optuna/{key}"
            if okey in state["heads"]:
                continue
            variant = rec["variant"]; seed = rec["seed"]; fold = key.split("/")[2]
            cond, bb = variant.split("_")
            run_id = rec["run_id"]
            tree = f"{ns.REMOTE_PROJECT_BASE}/outputs/symmetric_merged/native_en_text_heads_v1/{variant}_text_only/{run_id}/cv/{fold}"
            ckpt = f"{ns.REMOTE_PROJECT_BASE}/output_model/symmetric_merged/native_en_text_heads_v1/{variant}_text_only/{run_id}/cv/{fold}/best_model"
            mcfg = (
                f"/gpfs/projects/etur92/ozu647717/AudioLLM/experiment_runtime/"
                f"{slug}/configs/{run_id}/seed_{seed.lstrip('s')}/"
                f"symmetric_merged_text_heads_{variant}.yaml"
            )
            cmd = base() + [
                "submit-optuna100", slug,
                "--family", "merged", "--condition", cond, "--backbone", bb,
                "--run-name", run_id, "--fold", fold.lstrip("f"),
                "--seed", seed.lstrip("s"),
                "--features-dir", f"{tree}/features",
                "--merged-config", mcfg,
                "--stage", "cv", "--target-trials", "100",
                "--parent-checkpoint-path", ckpt,
                "--after-job-id", rec["post_job"],
                "--deployment-id", dep, "--scheduler-host", args.scheduler_host,
                "--execute",
            ]
            out = run(cmd, out_dir / (okey.replace("/", "_") + ".log"))
            jid = re.search(r"submitted optuna attempt job (\d+)", out).group(1)
            state["heads"][okey] = {"job": jid}
            submitted += 1
            if args.limit and submitted >= args.limit:
                break

    elif args.phase == "final":
        raise SystemExit(
            "final phase requires completed CV evidence; derive epochs first "
            "(rounded median of selected checkpoints) via submit-merged --stage final."
        )

    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
    counts = {p: len(state.get(p, {})) for p in ("chains", "heads", "final")}
    print(json.dumps({"submitted_this_run": submitted, **counts}, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
