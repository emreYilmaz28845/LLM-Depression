PRAGMA foreign_keys = ON;

CREATE TABLE schema_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE experiment_groups (
    group_id TEXT PRIMARY KEY,
    schema_version TEXT NOT NULL,
    title TEXT,
    research_question TEXT,
    github_issue INTEGER,
    github_pr INTEGER,
    status TEXT,
    source_path TEXT NOT NULL,
    source_sha256 TEXT NOT NULL
);

CREATE TABLE logical_runs (
    logical_run_id TEXT PRIMARY KEY,
    group_id TEXT REFERENCES experiment_groups(group_id),
    logical_run_name TEXT NOT NULL,
    dataset TEXT,
    modality TEXT,
    method TEXT,
    seed INTEGER
);

CREATE TABLE run_attempts (
    attempt_id TEXT PRIMARY KEY,
    logical_run_id TEXT NOT NULL REFERENCES logical_runs(logical_run_id),
    schema_version TEXT NOT NULL,
    legacy_import INTEGER NOT NULL CHECK (legacy_import IN (0, 1)),
    created_at_utc TEXT,
    git_commit TEXT,
    git_branch TEXT,
    git_dirty INTEGER CHECK (git_dirty IS NULL OR git_dirty IN (0, 1)),
    deployed_source_sha256 TEXT,
    resolved_config_sha256 TEXT,
    manifest_sha256 TEXT,
    split_sha256 TEXT,
    github_issue INTEGER,
    github_pr INTEGER,
    supersedes_attempt_id TEXT REFERENCES run_attempts(attempt_id),
    metadata_path TEXT,
    current_state TEXT NOT NULL
);

CREATE TABLE folds (
    fold_id INTEGER PRIMARY KEY,
    attempt_id TEXT NOT NULL REFERENCES run_attempts(attempt_id),
    fold INTEGER NOT NULL,
    run_dir TEXT NOT NULL,
    run_config_path TEXT,
    status_path TEXT,
    locally_verified INTEGER NOT NULL DEFAULT 0
        CHECK (locally_verified IN (0, 1)),
    UNIQUE (attempt_id, fold)
);

CREATE TABLE job_events (
    event_id TEXT PRIMARY KEY,
    fold_id INTEGER NOT NULL REFERENCES folds(fold_id),
    job_key TEXT NOT NULL,
    job_type TEXT NOT NULL,
    event_type TEXT NOT NULL,
    slurm_job_id TEXT,
    slurm_array_job_id TEXT,
    slurm_array_task_id TEXT,
    dependency_job_ids_json TEXT NOT NULL,
    status TEXT,
    at_utc TEXT NOT NULL,
    reason TEXT,
    resubmission_of_job_id TEXT
);

CREATE TABLE artifacts (
    artifact_id TEXT PRIMARY KEY,
    fold_id INTEGER NOT NULL REFERENCES folds(fold_id),
    artifact_type TEXT NOT NULL,
    role TEXT NOT NULL,
    path TEXT NOT NULL,
    sha256 TEXT,
    size_bytes INTEGER,
    exists_on_mn5 INTEGER CHECK (exists_on_mn5 IS NULL OR exists_on_mn5 IN (0, 1)),
    exists_locally INTEGER CHECK (exists_locally IS NULL OR exists_locally IN (0, 1)),
    locally_verified INTEGER NOT NULL DEFAULT 0
        CHECK (locally_verified IN (0, 1)),
    UNIQUE (fold_id, role, path)
);

CREATE TABLE evaluations (
    evaluation_id TEXT PRIMARY KEY,
    fold_id INTEGER NOT NULL REFERENCES folds(fold_id),
    dataset TEXT,
    split_name TEXT,
    split_protocol TEXT,
    checkpoint_role TEXT,
    checkpoint_path TEXT NOT NULL,
    backend TEXT,
    evaluation_view TEXT,
    aggregation TEXT,
    metric_namespace TEXT,
    metrics_artifact_id TEXT REFERENCES artifacts(artifact_id),
    predictions_artifact_id TEXT REFERENCES artifacts(artifact_id),
    locally_verified INTEGER NOT NULL DEFAULT 0
        CHECK (locally_verified IN (0, 1)),
    reportable INTEGER NOT NULL DEFAULT 0 CHECK (reportable IN (0, 1)),
    warnings_json TEXT NOT NULL
);

CREATE TABLE metrics (
    metric_id INTEGER PRIMARY KEY,
    evaluation_id TEXT NOT NULL REFERENCES evaluations(evaluation_id),
    namespace TEXT,
    metric_name TEXT NOT NULL,
    metric_value REAL,
    support INTEGER,
    aggregation TEXT,
    backend TEXT,
    evaluation_view TEXT,
    split_name TEXT,
    checkpoint_role TEXT,
    evidence_artifact_id TEXT REFERENCES artifacts(artifact_id),
    UNIQUE (
        evaluation_id, namespace, metric_name, aggregation, backend,
        evaluation_view, split_name, checkpoint_role
    )
);

CREATE TABLE provenance (
    provenance_id INTEGER PRIMARY KEY,
    attempt_id TEXT NOT NULL REFERENCES run_attempts(attempt_id),
    fold_id INTEGER NOT NULL REFERENCES folds(fold_id),
    key TEXT NOT NULL,
    value_json TEXT NOT NULL,
    source_artifact_id TEXT REFERENCES artifacts(artifact_id),
    UNIQUE (attempt_id, fold_id, key)
);

CREATE TABLE registry_imports (
    import_id INTEGER PRIMARY KEY,
    source_path TEXT NOT NULL,
    source_sha256 TEXT NOT NULL,
    importer_version TEXT NOT NULL,
    imported_at_utc TEXT NOT NULL,
    status TEXT NOT NULL,
    details_json TEXT NOT NULL,
    UNIQUE (source_path, source_sha256, importer_version)
);

CREATE INDEX idx_logical_runs_group ON logical_runs(group_id);
CREATE INDEX idx_logical_runs_dataset ON logical_runs(dataset);
CREATE INDEX idx_attempts_commit ON run_attempts(git_commit);
CREATE INDEX idx_attempts_state ON run_attempts(current_state);
CREATE INDEX idx_folds_attempt ON folds(attempt_id);
CREATE INDEX idx_jobs_fold_status ON job_events(fold_id, status);
CREATE INDEX idx_evaluations_qualified
    ON evaluations(dataset, backend, evaluation_view, aggregation, metric_namespace);
CREATE INDEX idx_metrics_lookup
    ON metrics(metric_name, namespace, backend, evaluation_view, aggregation);
CREATE INDEX idx_artifacts_fold ON artifacts(fold_id);

PRAGMA user_version = 1;
