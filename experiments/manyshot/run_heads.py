"""Fit the subject-mean L2 logistic head on frozen (arm A) and many-shot (arm B) caches on MN5.

Usage on an MN5 login node, venv gemma4_12b_tf5_14_1: python run_heads.py routes_native_no_turkish
"""
import json, subprocess, sys
from pathlib import Path
B = Path("/gpfs/projects/etur92/ozu647717/AudioLLM")
CODE = B / "deployments/feat-ozy-148-manyshot-embedding-20261116-20261116T212404Z-08a94a80-8b08ea56/code"
RT = B / "experiment_runtime/feat-ozy-148-manyshot-embedding-20261116"
MS = RT / "full/manyshot"
FROZEN = B / "experiment_runtime/exp-frozen-base-standalone-heads-20261024/caches"
routes = json.load(open(CODE / f"experiments/manyshot/{sys.argv[1]}.json"))
for r in routes:
    name = r["name"]; n = len(r["split_files"])
    for arm in ("A_frozen", "B_manyshot"):
        out = RT / "heads" / arm / name
        if out.exists():
            print("skip", out); continue
        if arm == "B_manyshot":
            caches = [str(MS / name / f"fold_{k}") for k in range(n)]
            dev = [str(MS / name / f"fold_{k}" / "dev") for k in range(n)] if r["protocol"] == "daic_official" else []
            extra = ["--per-fold-caches"]
        else:
            caches = [str(FROZEN / name / "main")]
            dev = [str(FROZEN / name / "dev")] if r["protocol"] == "daic_official" else []
            extra = []
        cmd = [sys.executable, str(CODE / "baselines/frozen_standalone_heads.py"), "--cache-dirs", *caches,
               *(["--dev-caches", *dev] if dev else []), *extra, "--split-files", *r["split_files"],
               "--protocol", r["protocol"], "--output-dir", str(out), "--route", json.dumps({"name": name, "arm": arm})]
        print(">>", arm, name, flush=True)
        subprocess.run(cmd, check=True, cwd=CODE, env={"PYTHONPATH": str(CODE), "PATH": "/usr/bin:/bin", "OMP_NUM_THREADS": "8"})
