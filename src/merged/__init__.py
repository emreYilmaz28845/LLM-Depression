"""Symmetric five-dataset experiment protocol.

The package intentionally sits beside the established single-dataset pipeline.
It owns the merged protocol's identities, splits, weighting, schedules, and
artifact contracts while reusing the existing component manifest and model
implementations.
"""

from .protocol import (
    DATASETS,
    METHODS,
    MODALITIES,
    build_dataset_aware_schedule,
    build_merged_manifest,
    build_protocol_splits,
    compute_hierarchical_example_weights,
    namespace_id,
)

__all__ = [
    "DATASETS",
    "METHODS",
    "MODALITIES",
    "build_dataset_aware_schedule",
    "build_merged_manifest",
    "build_protocol_splits",
    "compute_hierarchical_example_weights",
    "namespace_id",
]
