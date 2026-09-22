#!/usr/bin/env python3
"""Prepare the task-owned MN5 execution root for the merged prompt-context cells.

The managed workflow (`tools/exp.py deploy` / `exp submit`) covers standalone
backbone folds only; the merged campaign runs through
`scripts/submit_symmetric_merged.py`, which needs one PROJECT_ROOT that contains
both the code and the component manifests, and writes outputs below it.

This script therefore builds a *new*, task-owned execution root:

    <runtime>/<experiment_id>/merged/code           rsynced lane (tracked files)
    <runtime>/<experiment_id>/merged/code/outputs   component manifests + pooled manifest
    <runtime>/<experiment_id>/merged/code/output_model  merged runs and checkpoints

Nothing in the permanent cluster checkout is written. Component manifests are
copied read-only from the permanent checkout, and the pooled Turkish manifest is
built inside the new root from the audited `source_inputs_remote` of the earlier
pooled campaign. The script refuses to touch an existing non-empty root.

The same PROJECT_ROOT is used for every stage so the merged registry, protocol
artifacts, and fold outputs stay inside one task-owned tree.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

TRANSFER_HOST = "ozu647717@transfer1.bsc.es"
SCHEDULER_HOST = "ozu647717@alogin2.bsc.es"
REMOTE_RUNTIME_BASE = "/gpfs/projects/etur92/ozu647717/AudioLLM/experiment_runtime"
PERMANENT_CHECKOUT = "/gpfs/projects/etur92/ozu647717/AudioLLM/LLM-Depression"
QWEN_ENV_ACTIVATE = "/gpfs/projects/etur92/ozu647717/venvs/qwen_mn5_rebuilt/bin/activate"
# The pooled qcond campaign used the home-level runtime base, which is a
# different tree from the AudioLLM-level base used by the managed workflow.
POOLED_SOURCE_ROOT = (
    "/gpfs/projects/etur92/ozu647717/experiment_runtime/"
    "exp-turkish-pooled-qcond-clean-v1-20260903/source_inputs_remote"
)

# Component manifests reused read-only from the permanent checkout: the four
# non-Turkish datasets keep byte-identical manifests, splits and audio paths.
COPIED_COMPONENTS = {
    "daic": "daic/daic_manifest.jsonl",
    "d3tec": "d3tec/d3tec_manifest.jsonl",
    "cmdc": "cmdc/cmdc_manifest.jsonl",
    "androids_interview": "androids/androids_interview_manifest.jsonl",
}
COPIED_METADATA = {
    "daic": "daic/daic_manifest_metadata.json",
    "d3tec": "d3tec/d3tec_manifest_metadata.json",
    "cmdc": "cmdc/cmdc_manifest_metadata.json",
    "androids_interview": "androids/androids_interview_manifest_metadata.json",
}
MERGED_CONFIGS = {
    ("qwen", "audio_only"): "configs/experiments/merged/symmetric_merged_harmonized_promptcontext_audio_only.yaml",
    ("qwen", "text_only"): "configs/experiments/merged/symmetric_merged_harmonized_promptcontext_text_only.yaml",
    ("qwen", "audio_text"): "configs/experiments/merged/symmetric_merged_harmonized_promptcontext_audio_text.yaml",
    ("gemma4", "audio_only"): "configs/experiments/merged/symmetric_merged_harmonized_gemma4_promptcontext_audio_only.yaml",
    ("gemma4", "text_only"): "configs/experiments/merged/symmetric_merged_harmonized_gemma4_promptcontext_text_only.yaml",
    ("gemma4", "audio_text"): "configs/experiments/merged/symmetric_merged_harmonized_gemma4_promptcontext_audio_text.yaml",
}


class PrepError(RuntimeError):
    """Raised when the merged execution root cannot be prepared safely."""


def q(value: Any) -> str:
    return shlex.quote(str(value))


def ssh(host: str, script: str, *, dry_run: bool, timeout: int = 1800) -> str:
    header = "set -euo pipefail\n" + script
    if dry_run:
        print(f"--- ssh {host} <<'EOF'\n{header}EOF")
        return ""
    completed = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", host, "bash -s"],
        input=header,
        text=True,
        capture_output=True,
        timeout=timeout,
    )
    if completed.returncode != 0:
        raise PrepError(
            f"remote command failed on {host} (rc={completed.returncode}):\n"
            f"{completed.stdout}\n{completed.stderr}"
        )
    return completed.stdout.strip()


def rsync(source: str, destination: str, *, dry_run: bool) -> None:
    argv = [
        "rsync",
        "-avh",
        "--include=.provenance/***",
        "--filter=:- .gitignore",
        # A lane worktree stores .git as a file that points at the main
        # checkout, so the directory-only form would copy that pointer.
        "--exclude=.git",
    ]
    if dry_run:
        argv.append("--dry-run")
    argv += [f"{source}/", destination]
    print("$ " + " ".join(argv))
    completed = subprocess.run(argv, text=True, capture_output=True)
    if completed.returncode != 0:
        raise PrepError(f"rsync failed: {completed.stderr.strip()}")
    if dry_run:
        print(completed.stdout)


def build_script(remote_root: str) -> str:
    code = f"{remote_root}/code"
    manifests = f"{code}/outputs/manifests_harmonized"
    splits = f"{code}/outputs/splits_harmonized"
    lines = [
        "set -euo pipefail",
        "module purge",
        "module load bsc/1.0",
        "module load miniforge/24.3.0-0",
        f"source {q(QWEN_ENV_ACTIVATE)}",
        f"export PROJECT_ROOT={q(code)}",
        f"cd {q(code)}",
        f"pooled_source={q(POOLED_SOURCE_ROOT)}",
        f"permanent={q(PERMANENT_CHECKOUT)}",
    ]
    for dataset, relative in COPIED_COMPONENTS.items():
        source = f'"$permanent"/outputs/manifests_harmonized/{relative}'
        target = f"{manifests}/{relative}"
        lines += [
            f"mkdir -p $(dirname {q(target)})",
            f"test -f {source} || {{ echo 'missing component manifest:' {source} >&2; exit 1; }}",
            f"test ! -e {q(target)} || {{ echo 'refusing to overwrite:' {q(target)} >&2; exit 1; }}",
            f"cp {source} {q(target)}",
        ]
    for dataset, relative in COPIED_METADATA.items():
        source = f'"$permanent"/outputs/splits_harmonized/{relative}'
        target = f"{splits}/{relative}"
        lines += [
            f"mkdir -p $(dirname {q(target)})",
            f"test -f {source} || {{ echo 'missing split metadata:' {source} >&2; exit 1; }}",
            f"test ! -e {q(target)} || {{ echo 'refusing to overwrite:' {q(target)} >&2; exit 1; }}",
            f"cp {source} {q(target)}",
        ]
    lines += [
        f"test -d $permanent/.deps/qwen_hidden && cp -r $permanent/.deps {q(code)}/ || echo 'note: no .deps/qwen_hidden to copy'",
        "python scripts/build_turkish_pooled_manifest.py"
        f" --positive-native-manifest $pooled_source/manifests/pos_native/turkish_manifest.jsonl"
        f" --positive-native-split $pooled_source/splits/pos_native/turkish_folds.json"
        f" --negative-native-manifest $pooled_source/manifests/neg_native/turkish_manifest.jsonl"
        f" --negative-native-split $pooled_source/splits/neg_native/turkish_folds.json"
        f" --positive-english-manifest $pooled_source/manifests/pos_english/turkish_manifest.jsonl"
        f" --positive-english-split $pooled_source/splits/pos_english/turkish_folds.json"
        f" --negative-english-manifest $pooled_source/manifests/neg_english/turkish_manifest.jsonl"
        f" --negative-english-split $pooled_source/splits/neg_english/turkish_folds.json"
        f" --native-output-dir {manifests}/turkish_pooled_t17_qwen3asr"
        f" --native-split-output-dir {splits}/turkish_pooled_t17_qwen3asr"
        f" --english-output-dir {manifests}/turkish_pooled_t17_qwen3asr_en"
        f" --english-split-output-dir {splits}/turkish_pooled_t17_qwen3asr_en"
        f" --audit-output {code}/outputs/promptcontext_audit/pooled_manifest_audit.json",
    ]
    for (backend, modality), config in MERGED_CONFIGS.items():
        lines.append(
            f"python scripts/build_symmetric_merged_manifest.py --config {q(config)} "
            f"--skip-existing-components"
        )
    return "\n".join(lines) + "\n"


def audit_script(remote_root: str) -> str:
    code = f"{remote_root}/code"
    return f"""set -euo pipefail
python - <<'PY'
import hashlib, json, subprocess
from pathlib import Path
code = Path({code!r})
def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()
files = sorted(
    path for path in code.rglob('*')
    if path.is_file() and (
        path.name.endswith('_manifest.jsonl')
        or path.name.endswith('_manifest_metadata.json')
        or path.name.endswith('pooled_manifest_audit.json')
        or path.name == 'merged_protocol.json'
        or path.name == 'context.json'
    )
)
payload = {{
    'root': str(code),
    'hashes': {{str(path.relative_to(code)): sha(path) for path in files}},
}}
print(json.dumps(payload, indent=2, sort_keys=True))
PY
"""


def _local_git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=str(PROJECT_ROOT), text=True).strip()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", default="exp-prompt-context-qwen-gemma-v1-20260921")
    parser.add_argument(
        "--remote-root",
        default=None,
        help="defaults to <runtime base>/<experiment-id>/<root name>",
    )
    parser.add_argument(
        "--root-name",
        default=None,
        help="defaults to merged_<source commit 8>; a source change needs a new root",
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--source", default=str(PROJECT_ROOT))
    parser.add_argument("--audit-output", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root_name = args.root_name or f"merged_{_local_git('rev-parse', 'HEAD')[:8]}"
    remote_root = args.remote_root or f"{REMOTE_RUNTIME_BASE}/{args.experiment_id}/{root_name}"
    code = f"{remote_root}/code"
    dry_run = not args.execute
    print(f"merged execution root: {code} ({'dry-run' if dry_run else 'execute'})")

    if not dry_run:
        if _local_git("status", "--porcelain"):
            raise PrepError(
                "the merged code root must match one commit; commit or stash before preparing"
            )
        subprocess.run(["bash", str(PROJECT_ROOT / "scripts" / "capture_provenance.sh")], check=True)

    ssh(
        TRANSFER_HOST,
        f"test ! -e {q(remote_root)} || {{ echo 'refusing existing root {remote_root}' >&2; exit 1; }}\n"
        f"mkdir -p {q(code)}",
        dry_run=dry_run,
    )
    rsync(args.source, f"{TRANSFER_HOST}:{code}", dry_run=dry_run)
    print("--- build steps (scheduler login) ---")
    ssh(SCHEDULER_HOST, build_script(remote_root), dry_run=dry_run)
    print("--- audit ---")
    audit_json = ssh(TRANSFER_HOST, audit_script(remote_root), dry_run=dry_run)
    if not dry_run:
        payload = json.loads(audit_json)
        payload["source_commit"] = _local_git("rev-parse", "HEAD")
        payload["source_branch"] = _local_git("branch", "--show-current")
        payload["source_root"] = str(args.source)
        payload["source_clean"] = True
        output = args.audit_output or (PROJECT_ROOT / "outputs/promptcontext_audit/merged_prep.json")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(payload, indent=2, sort_keys=True))
        print(f"audit written: {output}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except PrepError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
