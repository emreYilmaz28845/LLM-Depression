from __future__ import annotations

SCHEMA_VERSION_METADATA = "audiollm.metadata.v1"
SCHEMA_VERSION_STATUS = "audiollm.status.v1"
SCHEMA_VERSION_JOB_EVENT = "audiollm.job_event.v1"
SCHEMA_VERSION_ARTIFACTS = "audiollm.artifacts.v1"
SCHEMA_VERSION_EVALUATIONS = "audiollm.evaluations.v1"
SCHEMA_VERSION_EXPERIMENT_GROUP = "audiollm.experiment_group.v1"
SCHEMA_VERSION_REPORT = "audiollm.report.v1"

SCHEMA_VERSIONS = (
    SCHEMA_VERSION_METADATA,
    SCHEMA_VERSION_STATUS,
    SCHEMA_VERSION_JOB_EVENT,
    SCHEMA_VERSION_ARTIFACTS,
    SCHEMA_VERSION_EVALUATIONS,
    SCHEMA_VERSION_EXPERIMENT_GROUP,
    SCHEMA_VERSION_REPORT,
)

LIFECYCLE_STATES = (
    "PLANNED",
    "IMPORTED_LEGACY",
    "DEPLOYED",
    "SUBMITTED",
    "RUNNING",
    "COMPLETED_ON_MN5",
    "FAILED",
    "CANCELLED",
    "SYNCED_LOCALLY",
    "LOCALLY_VALIDATED",
    "REPORTABLE",
    "SUPERSEDED",
)

ARTIFACT_TYPES = (
    "run_config",
    "source_manifest",
    "manifest",
    "split",
    "checkpoint",
    "metrics",
    "predictions",
    "audit",
    "training_history",
    "summary",
    "report",
    "wandb_offline",
)

JOB_TYPES = (
    "train",
    "evaluation",
    "summary",
    "audit",
    "collect",
    "train_eval",
    "hidden_classifier",
)

JOB_EVENT_TYPES = (
    "SUBMITTED",
    "STARTED",
    "COMPLETED",
    "FAILED",
    "CANCELLED",
    "OBSERVED",
)

JOB_STATUS_VALUES = (
    "PENDING",
    "RUNNING",
    "COMPLETED",
    "FAILED",
    "CANCELLED",
    "TIMEOUT",
    "NODE_FAIL",
    "OUT_OF_MEMORY",
    "PREEMPTED",
)

CHECKPOINT_ROLES = ("best_model", "last_model")

WANDB_PROJECT = "audiollm-depression"

WANDB_SYNC_STATUSES = ("NOT_EXPORTED", "OFFLINE", "SYNCED", "INCOMPLETE")

LEGACY_ATTEMPT_ID_ALGORITHM_VERSION = "legacy-attempt-v1"

ATTEMPT_ID_SUFFIX_HEX_LENGTH = 8
