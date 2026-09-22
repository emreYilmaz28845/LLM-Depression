"""Manifest build policy for managed submissions.

Normal datasets build their manifest inside the worker before training starts.
The pooled Turkish dataset is different: its manifest is built from the two
source manifests by ``scripts/build_turkish_pooled_manifest.py`` outside the
worker, and the worker's own builder cannot reconstruct it. Such a submission
declares ``manifest_policy: prebuilt``, and the submit path then skips the
worker-side build and verifies the prebuilt files before any job is queued.

Defaults preserve the existing behaviour: a config without ``manifest_policy``
keeps ``build``.
"""

from __future__ import annotations

from typing import Any

MANIFEST_POLICY_BUILD = "build"
MANIFEST_POLICY_PREBUILT = "prebuilt"
MANIFEST_POLICIES = (MANIFEST_POLICY_BUILD, MANIFEST_POLICY_PREBUILT)
POOLED_DATASET_VARIANT = "pooled_t17"


class ManifestPolicyError(ValueError):
    """Raised when a manifest policy is unknown or cannot be honoured."""


def resolve_manifest_policy(config: dict[str, Any], override: str | None = None) -> str:
    """Return the effective manifest policy for a resolved config."""
    raw_value = override if override is not None else config.get("manifest_policy")
    if raw_value is None or str(raw_value).strip() == "":
        return MANIFEST_POLICY_BUILD
    policy = str(raw_value).strip().lower()
    if policy not in MANIFEST_POLICIES:
        raise ManifestPolicyError(
            f"Unsupported manifest policy {raw_value!r}. "
            f"Expected one of {list(MANIFEST_POLICIES)}."
        )
    return policy


def validate_manifest_policy(config: dict[str, Any], override: str | None = None) -> str:
    """Fail closed when the pooled recipe would be sent down the build route."""
    policy = resolve_manifest_policy(config, override)
    dataset_variant = str(config.get("dataset_variant", "")).strip()
    if policy == MANIFEST_POLICY_BUILD and dataset_variant == POOLED_DATASET_VARIANT:
        raise ManifestPolicyError(
            f"dataset_variant={POOLED_DATASET_VARIANT} requires "
            f"manifest_policy={MANIFEST_POLICY_PREBUILT}: the pooled manifest is "
            "built by scripts/build_turkish_pooled_manifest.py, and the worker "
            "must not rebuild it."
        )
    return policy


def prebuilt_manifest_files(
    *, manifest_dir: str, split_dir: str, dataset: str
) -> dict[str, str]:
    """Paths a prebuilt submission must already provide.

    The caller passes the directories the workers will actually read, so the
    verification checks the same files the run resolves.
    """
    manifest_dir = str(manifest_dir).rstrip("/")
    split_dir = str(split_dir).rstrip("/")
    dataset = str(dataset)
    return {
        "manifest": f"{manifest_dir}/{dataset}_manifest.jsonl",
        "manifest_csv": f"{manifest_dir}/{dataset}_manifest.csv",
        "folds": f"{split_dir}/{dataset}_folds.json",
        "split_metadata": f"{split_dir}/{dataset}_manifest_metadata.json",
    }
