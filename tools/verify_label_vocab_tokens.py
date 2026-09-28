#!/usr/bin/env python3
"""Backend-aware answer-label token audit for the Qwen3 DAIC label-vocabulary campaign.

``tools/verify_ab_label_tokens.py`` is bound to the Qwen2 prompt skeleton and
rejects the Qwen3.8 and Qwen3-Omni backends, so it cannot check the campaign's
prompt boundary. This audit instead uses the real backend processor loader
(``src.model.runtime.load_processor``) and the real example-preparation path
(``_base_example_from_row`` for synthetic rows, ``build_examples`` for real
manifest rows, then ``prepare_backend_examples``), which is the same rendering
training and likelihood scoring use. It loads no model weights.

For every config it checks, on a synthetic example and on real DAIC train/val
rows when a manifest is supplied:

* the two internal class labels tokenize to a non-empty continuation that starts
  exactly at the prompt boundary, and the prompt token prefix is unchanged;
* each of the four short vocabularies (ab, 01, truefalse, yesno) uses exactly one
  complete continuation token, and the two classes never share a token id;
* the training label mask keeps exactly the candidate continuation tokens (the
  chat terminator is not counted as the answer token);
* the gold label never enters the inference prompt.

The English vocabulary may span several tokens; its token lengths are recorded
and the mean-token log-probability definition is unchanged. Failures are reported
with evidence and never fixed by silently changing case, spacing or wording.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.runtime import (  # noqa: E402
    _base_example_from_row,
    build_examples,
    resolve_audio_placeholder,
)
from src.model.runtime import build_collator, load_processor, prepare_backend_examples  # noqa: E402
from src.utils import (  # noqa: E402
    INPUT_MODALITY_TEXT_ONLY,
    external_label_text_from_int,
    internal_label_text_from_int,
    load_yaml_with_overrides,
    resolve_input_modality,
    resolve_label_config,
    resolve_model_backend,
    resolve_project_path,
)

SCHEMA_VERSION = "audiollm.qwen3_daic_label_token_audit.v1"
SUPPORTED_BACKENDS = ("qwen38", "qwen3omni")
SINGLE_TOKEN_ARMS = ("ab", "01", "truefalse", "yesno")
AUDITABLE_MODALITIES = ("text_only", "audio_only", "audio_text")
SYNTHETIC_TRANSCRIPT = "Synthetic audit transcript. Not real participant data."
ARM_BY_VOCAB = {
    "short_internal_ab_labels": "ab",
    "binary_01_labels": "01",
    "truefalse_labels": "truefalse",
    "yesno_labels": "yesno",
    "legacy_english_labels": "en",
}


class AuditError(RuntimeError):
    """Raised when the audit cannot run at all (fail closed)."""


def _token_ids(processor, text: str) -> list[int]:
    encoded = processor(text=text, return_tensors=None, padding=False)
    ids = encoded["input_ids"]
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return [int(token_id) for token_id in ids]


def _decode(processor, token_ids: list[int]) -> str:
    tokenizer = getattr(processor, "tokenizer", processor)
    return tokenizer.decode(token_ids)


def _terminator_token_ids(processor, prompt_text: str, training_text: str, label_text: str) -> list[int]:
    if not training_text.startswith(prompt_text):
        raise AuditError("training text does not start with the prompt text")
    suffix = training_text[len(prompt_text) :]
    if not suffix.startswith(label_text):
        raise AuditError("training text does not place the label directly after the prompt")
    terminator = suffix[len(label_text) :]
    if not terminator:
        raise AuditError("training text has no chat terminator after the label")
    return _token_ids(processor, terminator)


def _synthetic_row(config: dict[str, Any], *, label: int) -> dict[str, Any]:
    return {
        "dataset": str(config["dataset"]),
        "subject_id": "audit-subject",
        "sample_id": "audit-synthetic",
        "transcript": SYNTHETIC_TRANSCRIPT,
        "label": int(label),
        "label_text": external_label_text_from_int(config, int(label)),
        "audio_path": "audit-synthetic.wav",
        "start_time": 0.0,
        "end_time": 30.0,
    }


def build_synthetic_example(config: dict[str, Any], processor, *, label: int) -> dict[str, Any]:
    max_chars = int(config.get("data", {}).get("transcript_max_chars", 0) or 0)
    example, _ = _base_example_from_row(_synthetic_row(config, label=label), config, max_chars)
    return prepare_backend_examples([example], config, processor)[0]


def _example_checks(
    config: dict[str, Any],
    processor,
    example: dict[str, Any],
    *,
    arm: str,
    failures: list[str],
    source: str,
) -> dict[str, Any]:
    tokenizer = getattr(processor, "tokenizer", processor)
    prompt_text = example["prompt_text"]
    training_text = example["training_text"]
    prompt_ids = _token_ids(processor, prompt_text)

    result: dict[str, Any] = {"source": source, "prompt_tokens": len(prompt_ids), "labels": {}}
    continuation_by_label: dict[int, list[int]] = {}
    for label in (1, 0):
        candidate = internal_label_text_from_int(config, label)
        full_ids = _token_ids(processor, prompt_text + candidate)
        if full_ids[: len(prompt_ids)] != prompt_ids:
            failures.append(f"{source}: candidate {candidate!r} changes the prompt token prefix")
        continuation = full_ids[len(prompt_ids) :]
        continuation_by_label[label] = continuation
        if not continuation:
            failures.append(f"{source}: candidate {candidate!r} has an empty continuation")
        elif _decode(processor, continuation).strip() != candidate:
            failures.append(
                f"{source}: candidate {candidate!r} does not round-trip from its continuation tokens"
            )
        if arm in SINGLE_TOKEN_ARMS and len(continuation) != 1:
            failures.append(
                f"{source}: candidate {candidate!r} uses {len(continuation)} tokens at the prompt "
                "boundary; the short vocabularies require exactly one"
            )
        result["labels"][label] = {
            "text": candidate,
            "tokens": continuation,
            "length": len(continuation),
        }
    if continuation_by_label[1] == continuation_by_label[0]:
        failures.append(f"{source}: the two classes share the same continuation token ids")

    terminator_ids = _terminator_token_ids(processor, prompt_text, training_text, example["internal_label_text"])
    result["terminator_tokens"] = len(terminator_ids)
    for label, continuation in continuation_by_label.items():
        for token_id in continuation:
            if token_id in {getattr(tokenizer, "eos_token_id", None), getattr(tokenizer, "pad_token_id", None)}:
                failures.append(
                    f"{source}: answer token {token_id} for label {label} is the EOS/padding token"
                )

    collator = build_collator(config, processor, debug=True)
    prepared = dict(example)
    prepared["audio_arrays"] = []
    collator([prepared])
    debug = collator.last_debug_example or {}
    training_ids = debug.get("input_ids", [])
    unmasked = list(debug.get("unmasked_token_ids", []))
    if training_ids[: len(prompt_ids)] != prompt_ids:
        failures.append(f"{source}: the training sequence does not start with the exact prompt tokens")
    expected = continuation_by_label[int(example["label"])]
    if unmasked[: len(expected)] != expected:
        failures.append(
            f"{source}: the training label mask does not keep the candidate continuation tokens"
        )
    if len(unmasked) != len(expected) + len(terminator_ids):
        failures.append(
            f"{source}: the unmasked label span is {len(unmasked)} tokens, expected "
            f"{len(expected)} answer tokens plus {len(terminator_ids)} terminator tokens"
        )
    result["mask"] = {"unmasked_tokens": len(unmasked), "answer_tokens": len(expected)}
    return result


def _load_real_examples(
    config: dict[str, Any],
    processor,
    *,
    manifest: str,
    split_metadata: str | None,
    partitions: tuple[str, ...],
    limit: int,
) -> dict[str, list[dict[str, Any]]]:
    rows = [
        json.loads(line)
        for line in Path(manifest).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise AuditError(f"manifest has no rows: {manifest}")
    subject_partitions: dict[str, set[str]] = {}
    if split_metadata:
        payload = json.loads(Path(split_metadata).read_text(encoding="utf-8"))
        records = payload["subject_partitions"] if isinstance(payload, dict) else payload
        for record in records:
            subject_partitions.setdefault(str(record["partition"]), set()).add(str(record["subject_id"]))

    examples: dict[str, list[dict[str, Any]]] = {}
    for partition in partitions:
        if subject_partitions:
            subjects = sorted(subject_partitions.get(partition, set()))[:limit]
            selected = [row for row in rows if str(row["subject_id"]) in set(subjects)]
        else:
            candidates = [row for row in rows if str(row.get("split_original", "")) == partition]
            subjects = sorted({str(row["subject_id"]) for row in candidates})[:limit]
            selected = [row for row in candidates if str(row["subject_id"]) in set(subjects)]
        first_row_per_subject: dict[str, dict[str, Any]] = {}
        for row in selected:
            first_row_per_subject.setdefault(str(row["subject_id"]), row)
        chosen = [first_row_per_subject[key] for key in sorted(first_row_per_subject)][:limit]
        if not chosen:
            raise AuditError(f"no real {partition!r} rows found for the audit")
        built = build_examples(chosen, config, partition_name=partition)
        examples[partition] = prepare_backend_examples(built, config, processor)
    return examples


def audit_config(
    config_path: Path,
    *,
    model_path: str | None,
    overrides: list[str],
    manifest: str | None,
    split_metadata: str | None,
    partitions: tuple[str, ...],
    real_limit: int,
) -> dict[str, Any]:
    config = load_yaml_with_overrides(config_path, list(overrides))
    labels = resolve_label_config(config)
    arm = ARM_BY_VOCAB.get(str(labels["label_vocab_version"]))
    if arm is None:
        raise AuditError(f"{config_path}: unsupported label vocabulary {labels['label_vocab_version']!r}")
    modality = resolve_input_modality(config)
    if modality not in AUDITABLE_MODALITIES:
        raise AuditError(f"{config_path}: unsupported modality {modality!r}")
    backend = resolve_model_backend(config)
    if backend not in SUPPORTED_BACKENDS:
        raise AuditError(f"{config_path}: unsupported backend {backend!r} for this audit")
    if backend == "qwen38" and modality != INPUT_MODALITY_TEXT_ONLY:
        raise AuditError(f"{config_path}: the qwen38 backend is text-only")
    if backend == "qwen3omni" and modality == INPUT_MODALITY_TEXT_ONLY:
        raise AuditError(f"{config_path}: qwen3omni configs in this campaign are audio cells only")

    resolved_model = model_path or resolve_project_path(config["model_name_or_path"])
    if not Path(resolved_model).exists():
        raise AuditError(f"{config_path}: model snapshot not found: {resolved_model}")
    processor = load_processor(resolved_model, config)

    failures: list[str] = []
    positives = build_synthetic_example(config, processor, label=1)
    negatives = build_synthetic_example(config, processor, label=0)
    checks: dict[str, Any] = {}
    checks["prompt_is_gold_independent"] = positives["prompt_text"] == negatives["prompt_text"]
    if not checks["prompt_is_gold_independent"]:
        failures.append("synthetic: the prompt changes with the gold label")
    for label, example in ((1, positives), (0, negatives)):
        candidate = internal_label_text_from_int(config, label)
        if example["prompt_text"].rstrip().endswith(candidate):
            failures.append(f"synthetic: the prompt ends with the gold label text {candidate!r}")
    checks["audio_placeholder"] = resolve_audio_placeholder(config)

    synthetic_reports = [
        _example_checks(config, processor, positives, arm=arm, failures=failures, source="synthetic-label-1"),
        _example_checks(config, processor, negatives, arm=arm, failures=failures, source="synthetic-label-0"),
    ]

    real_reports: dict[str, list[dict[str, Any]]] = {}
    if manifest:
        real_examples = _load_real_examples(
            config,
            processor,
            manifest=manifest,
            split_metadata=split_metadata,
            partitions=partitions,
            limit=real_limit,
        )
        for partition, examples in real_examples.items():
            real_reports[partition] = [
                _example_checks(
                    config,
                    processor,
                    example,
                    arm=arm,
                    failures=failures,
                    source=f"real-{partition}-{index}",
                )
                for index, example in enumerate(examples)
            ]

    tokenizer = getattr(processor, "tokenizer", processor)
    tokenizer_json = Path(resolved_model) / "tokenizer.json"
    report: dict[str, Any] = {
        "config": str(config_path.relative_to(ROOT)) if str(config_path).startswith(str(ROOT)) else str(config_path),
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "arm": arm,
        "label_vocab_version": labels["label_vocab_version"],
        "internal_positive_label": labels["internal_positive_label"],
        "internal_negative_label": labels["internal_negative_label"],
        "dataset": config["dataset"],
        "modality": modality,
        "backend": backend,
        "model_path": str(resolved_model),
        "model_path_source": "cli" if model_path else "config",
        "tokenizer": {
            "class": type(tokenizer).__name__,
            "sha256_of_tokenizer_json": (
                hashlib.sha256(tokenizer_json.read_bytes()).hexdigest() if tokenizer_json.is_file() else None
            ),
        },
        "single_token_required": arm in SINGLE_TOKEN_ARMS,
        "checks": checks,
        "synthetic": synthetic_reports,
        "real": real_reports,
        "failures": failures,
        "passed": not failures,
    }
    return report


def default_configs(modalities: tuple[str, ...]) -> list[Path]:
    from scripts.build_qwen3_daic_label_configs import ARM_ORDER, SOURCES, generated_name

    paths: list[Path] = []
    for modality in SOURCES:
        if modalities and modality not in modalities:
            continue
        for arm in ARM_ORDER:
            paths.append(ROOT / "configs/labels" / generated_name(modality, arm))
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", action="append", default=[], help="config path (repeatable)")
    parser.add_argument("--modality", action="append", default=[], choices=AUDITABLE_MODALITIES)
    parser.add_argument("--model-path", default=None, help="override the config model path")
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    parser.add_argument("--manifest", default=None, help="DAIC manifest JSONL with real rows")
    parser.add_argument("--split-metadata", default=None, help="DAIC subject partition JSON")
    parser.add_argument("--partition", action="append", default=[], help="real partitions to audit (repeatable)")
    parser.add_argument("--real-limit", type=int, default=3)
    parser.add_argument("--output", default=None)
    parser.add_argument("--require-real-examples", action="store_true")
    args = parser.parse_args(argv)

    partitions = tuple(args.partition) or ("train",)
    config_paths = [Path(path) for path in args.config] or default_configs(tuple(args.modality))
    if not config_paths:
        raise AuditError("no configs selected for the audit")

    entries: list[dict[str, Any]] = []
    failures: list[str] = []
    for path in config_paths:
        report = audit_config(
            path,
            model_path=args.model_path,
            overrides=args.overrides,
            manifest=args.manifest,
            split_metadata=args.split_metadata,
            partitions=partitions,
            real_limit=args.real_limit,
        )
        if args.require_real_examples and not report["real"]:
            report["failures"].append("no real manifest examples were audited")
            report["passed"] = False
        entries.append(report)
        failures.extend(f"{report['config']}: {message}" for message in report["failures"])

    payload = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "environment": {
            "python": platform.python_version(),
            "transformers": _version("transformers"),
            "torch": _version("torch"),
        },
        "partitions": list(partitions),
        "manifest": args.manifest,
        "configs": entries,
        "failures": failures,
        "passed": not failures,
    }
    if args.output:
        Path(args.output).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    for entry in entries:
        status = "PASS" if entry["passed"] else "FAIL"
        lengths = {item["text"]: item["length"] for item in entry["synthetic"][0]["labels"].values()}
        real = {key: len(value) for key, value in entry["real"].items()}
        print(f"{status} {entry['config']} arm={entry['arm']} backend={entry['backend']} tokens={lengths} real={real}")
    if failures:
        print("\nfailures:")
        for message in failures:
            print(f"- {message}")
        return 1
    print(f"token audit passed for {len(entries)} configs")
    return 0


def _version(module_name: str) -> str | None:
    try:
        module = __import__(module_name)
    except Exception:  # pragma: no cover - version reporting only
        return None
    return getattr(module, "__version__", None)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AuditError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)
