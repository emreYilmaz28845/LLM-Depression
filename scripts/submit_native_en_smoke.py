#!/usr/bin/env python3
"""Reconcile and submit the locked 32-job smoke matrix via managed commands.

Dry-run first (default): every managed command runs with --dry-run and the
driver writes an expansion manifest asserting the locked counts.

    python scripts/submit_native_en_smoke.py --deployment-id <id>          # reconcile
    python scripts/submit_native_en_smoke.py --deployment-id <id> --execute

Expansion contract (execution-plan section 14):
  standalone: 4 panels (native/en x qwen/gemma4) x
              (train+eval chain=2, logreg attempt=1, optuna two-trial=1) = 16
  merged:     4 variants x (train+postprocess+head chain=3,
              optuna two-trial=1)                                      = 16
  total                                                                 = 32
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
SEED_SMOKE = 1337

PANEL_DATASET = {
    "native-qwen": "d3tec",
    "english-qwen": "androids_interview",
    "native-gemma4": "cmdc",
    "english-gemma4": "turkish",
}
PANEL_CONFIG = {
    "native-qwen": "configs/main/d3tec_text_only_harmonized_selmacrof1_tf.yaml",
    "english-qwen": "configs/main/androids_text_only_harmonized_selmacrof1_tf_en.yaml",
    "native-gemma4": "configs/main/cmdc_text_only_harmonized_selmacrof1_tf_gemma4_12b.yaml",
    "english-gemma4": "configs/main/turkish_t17_text_only_harmonized_selmacrof1_tf_qwen3asr_en_gemma4_12b.yaml",
}
MERGED_VARIANTS = ("native_qwen", "english_qwen", "native_gemma4", "english_gemma4")

EXPECTED_JOBS = 32


def _run(cmd: list[str], log_path: Path) -> str:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    log_path.write_text(proc.stdout + ("\n[stderr]\n" + proc.stderr if proc.stderr else ""))
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout).strip().splitlines()[-5:]
        raise RuntimeError(f"command failed ({' '.join(cmd[:4])}…): " + " | ".join(tail))
    return proc.stdout


def _parse_chain_ids(stdout: str) -> dict[str, str]:
    matches = re.findall(r"submitted jobs: (\{[^\n]*\})", stdout)
    if not matches:
        raise RuntimeError("chain output lacks 'submitted jobs: {...}'")
    return json.loads(matches[-1].replace("'", '"'))


def _parse_merged_fold_ids(stdout: str) -> dict[str, str]:
    matches = re.findall(r"fold (\d+) jobs: (\{[^\n]*\})", stdout)
    if not matches:
        raise RuntimeError("merged output lacks 'fold N jobs: {...}'")
    fold, payload = matches[-1]
    return {"fold": fold, **json.loads(payload.replace("'", '"'))}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slug", default=SLUG_DEFAULT)
    parser.add_argument("--deployment-id", required=True)
    parser.add_argument("--scheduler-host", default=SCHEDULER_DEFAULT)
    parser.add_argument("--seed", type=int, default=SEED_SMOKE)
    parser.add_argument("--out-root", default=None)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--skip-standalone", action="store_true",
                        help="resume mode: standalone panels already submitted")
    parser.add_argument("--panels", default=None,
                        help="comma list of panel keys to include (default all)")
    parser.add_argument("--variants", default=None,
                        help="comma list of merged variants to include (default all)")
    parser.add_argument("--name-suffix", default="",
                        help="appended to run names / run ids for retry identities")
    args = parser.parse_args()

    mode_dir = "exec" if args.execute else "dry"
    out_root = (
        Path(args.out_root)
        if args.out_root
        else PROJECT_ROOT / f"outputs/native_en_text_heads_smoke/{args.deployment_id[-16:]}/{mode_dir}"
    )
    out_root.mkdir(parents=True, exist_ok=True)

    jobs: list[dict] = []

    def base() -> list[str]:
        return [
            sys.executable, str(PROJECT_ROOT / "tools/exp.py"),
            "--db", str(PROJECT_ROOT / "outputs/experiment_registry/experiments.sqlite"),
        ]

    def submit(cmd: list[str], tag: str) -> str:
        mode = "--execute" if args.execute else "--dry-run"
        print(f"[{mode}] {tag}", flush=True)
        return _run(cmd + [mode], out_root / f"{tag}.log")

    # --- standalone panels -------------------------------------------------
    selected_panels = (
        [(k, PANEL_DATASET[k]) for k in PANEL_DATASET if k in set((args.panels or "").split(","))]
        if args.panels else list(PANEL_DATASET.items())
    )
    if args.skip_standalone:
        selected_panels = []
    for panel, dataset in selected_panels:
        cond, bb = panel.split("-")
        campaign = ns.campaign_for(cond, bb)
        run_name = f"smoke-{panel}-{dataset}{args.name_suffix}"
        chain_cmd = base() + [
            "submit", args.slug,
            "--config", PANEL_CONFIG[panel],
            "--fold", "0", "--seed", str(args.seed),
            "--run-name", run_name,
            "--campaign", campaign, "--modality", "text_only",
            "--dataset", dataset,
            "--deployment-id", args.deployment_id,
            "--scheduler-host", args.scheduler_host,
            "--set", "training.num_train_epochs=1",
            "--set", "split.smoke_subject_limit=6",
        ]
        if bb == "qwen":
            chain_cmd += ["--set", "evaluation.evaluation_view=harmonized_all_windows_full_coverage"]
        stdout = submit(chain_cmd, f"{panel}-chain")
        entry = {
            "panel": panel, "kind": "standalone_chain", "jobs_expected": 2,
            "run_name": run_name,
        }
        if args.execute:
            ids = _parse_chain_ids(stdout)
            entry["job_ids"] = ids
        jobs.append(entry)

        parent_fold_dir = (
            f"{ns.REMOTE_PROJECT_BASE}/output_model/{campaign}/text_only/{dataset}/{run_name}/fold_0"
        )
        hidden_cmd = base() + [
            "submit-hidden", args.slug,
            "--parent-fold-dir", parent_fold_dir,
            "--dataset", dataset, "--condition", cond, "--backbone", bb,
            "--run-name", run_name, "--fold", "0", "--seed", str(args.seed),
            "--deployment-id", args.deployment_id,
            "--scheduler-host", args.scheduler_host,
        ]
        if args.execute:
            hidden_cmd += [
                "--after-job-id", entry["job_ids"]["best_eval"],
            ]
        stdout = submit(hidden_cmd, f"{panel}-hidden-logreg")
        hentry = {"panel": panel, "kind": "logreg_attempt", "jobs_expected": 1}
        if args.execute:
            m = re.search(r"submitted logreg attempt job (\d+)", stdout)
            hentry["job_id"] = m.group(1)
        jobs.append(hentry)

        optuna_cmd = base() + [
            "submit-optuna100", args.slug,
            "--family", "standalone",
            "--condition", cond, "--backbone", bb,
            "--dataset", dataset,
            "--run-name", run_name, "--fold", "0", "--seed", str(args.seed),
            "--stage", "smoke", "--target-trials", "2",
            "--parent-checkpoint-path", f"{parent_fold_dir}/best_model",
            "--deployment-id", args.deployment_id,
            "--scheduler-host", args.scheduler_host,
        ]
        if args.execute:
            # Features appear when the hidden/logreg job finishes extraction.
            optuna_cmd += ["--after-job-id", hentry["job_id"]]
        stdout = submit(optuna_cmd, f"{panel}-optuna2")
        oentry = {"panel": panel, "kind": "optuna_two_trial", "jobs_expected": 1}
        if args.execute:
            m = re.search(r"submitted optuna attempt job (\d+)", stdout)
            oentry["job_id"] = m.group(1)
        jobs.append(oentry)

    # --- merged variants ---------------------------------------------------
    selected_variants = (
        [v for v in MERGED_VARIANTS if v in set((args.variants or "").split(","))]
        if args.variants else list(MERGED_VARIANTS)
    )
    for variant in selected_variants:
        cond, bb = variant.split("_")
        run_id = f"smoke-tmh-{variant}{args.name_suffix}"
        merged_cfg_local = (
            PROJECT_ROOT / f"configs/experiments/merged/symmetric_merged_text_heads_{variant}.yaml"
        )
        merged_cmd = base() + [
            "submit-merged", args.slug,
            "--config", str(merged_cfg_local),
            "--stage", "smoke", "--run-id", run_id,
            "--seed", str(args.seed),
            "--condition", cond, "--backbone", bb,
            "--folds", "0",
            "--subjects-per-class", "2",
            "--deployment-id", args.deployment_id,
            "--scheduler-host", args.scheduler_host,
        ]
        stdout = submit(merged_cmd, f"{variant}-merged-chain")
        mentry = {"variant": variant, "kind": "merged_chain", "jobs_expected": 3}
        derived_cfg_remote = ""
        if args.execute:
            ids = _parse_merged_fold_ids(stdout)
            mentry["job_ids"] = {k: ids[k] for k in ("train", "postprocess", "head")}
            derived_cfg_remote = (
                f"/gpfs/projects/etur92/ozu647717/AudioLLM/experiment_runtime/"
                f"{args.slug}/configs/{run_id}/seed_{args.seed}/{merged_cfg_local.name}"
            )
        jobs.append(mentry)

        features_dir = (
            f"{ns.REMOTE_PROJECT_BASE}/outputs/symmetric_merged/"
            f"native_en_text_heads_v1/{variant}_text_only/{run_id}/smoke/fold_0/features"
        )
        checkpoint_dir = (
            f"{ns.REMOTE_PROJECT_BASE}/output_model/symmetric_merged/"
            f"native_en_text_heads_v1/{variant}_text_only/{run_id}/smoke/fold_0/best_model"
        )
        optuna_cmd = base() + [
            "submit-optuna100", args.slug,
            "--family", "merged",
            "--condition", cond, "--backbone", bb,
            "--run-name", run_id, "--fold", "0", "--seed", str(args.seed),
            "--features-dir", features_dir,
            "--parent-checkpoint-path", checkpoint_dir,
        ]
        if args.execute:
            optuna_cmd += [
                "--merged-config", derived_cfg_remote,
                "--stage", "smoke",
                "--target-trials", "2",
                "--after-job-id", mentry["job_ids"]["postprocess"],
            ]
        else:
            optuna_cmd += ["--stage", "smoke", "--target-trials", "2"]
        stdout = submit(optuna_cmd, f"{variant}-optuna2")
        oentry = {"variant": variant, "kind": "optuna_two_trial", "jobs_expected": 1}
        if args.execute:
            m = re.search(r"submitted optuna attempt job (\d+)", stdout)
            oentry["job_id"] = m.group(1)
        jobs.append(oentry)

    expected_total = sum(j["jobs_expected"] for j in jobs)
    locked_expected = EXPECTED_JOBS if not (args.skip_standalone or args.panels or args.variants) else expected_total
    reconciliation = {
        "schema_version": "audiollm.native_en_smoke_reconciliation.v1",
        "deployment_id": args.deployment_id,
        "mode": "execute" if args.execute else "dry_run",
        "entries": jobs,
        "expected_jobs": expected_total,
        "locked_jobs": EXPECTED_JOBS,
        "match": expected_total == locked_expected,
    }
    recon_path = out_root / "reconciliation.json"
    recon_path.write_text(json.dumps(reconciliation, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: reconciliation[k] for k in ("expected_jobs", "locked_jobs", "match")}, indent=2))
    if not reconciliation["match"]:
        print(f"ERROR: expansion {expected_total} != locked {EXPECTED_JOBS}", file=sys.stderr)
        return 1
    if not args.execute:
        print("dry-run reconciliation passed; rerun with --execute to submit")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
