from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import re
from pathlib import Path
from typing import Any

import numpy as np
import yaml


LABEL_DEPRESSED = "Depressed"
LABEL_NON_DEPRESSED = "Non-depressed"
LABEL_TEXT_BY_INT = {0: LABEL_NON_DEPRESSED, 1: LABEL_DEPRESSED}
LABEL_INT_BY_TEXT = {value: key for key, value in LABEL_TEXT_BY_INT.items()}
LABEL_VOCAB_VERSION_LEGACY = "legacy_english_labels"
LABEL_VOCAB_VERSION_SHORT_AB = "short_internal_ab_labels"
SUPPORTED_LABEL_VOCAB_VERSIONS = (LABEL_VOCAB_VERSION_LEGACY, LABEL_VOCAB_VERSION_SHORT_AB)
INPUT_MODALITY_AUDIO_TEXT = "audio_text"
INPUT_MODALITY_AUDIO_ONLY = "audio_only"
INPUT_MODALITY_TEXT_ONLY = "text_only"
SUPPORTED_INPUT_MODALITIES = (
    INPUT_MODALITY_AUDIO_TEXT,
    INPUT_MODALITY_AUDIO_ONLY,
    INPUT_MODALITY_TEXT_ONLY,
)
PREDICTION_MODE_LIKELIHOOD = "likelihood"
PREDICTION_MODE_GENERATION = "generation"
PREDICTION_MODE_ORIGINAL_TEACHER_FORCED = "original_teacher_forced"
PREDICTION_MODE_PROTOCOLS = {
    PREDICTION_MODE_LIKELIHOOD: "candidate_label_likelihood",
    PREDICTION_MODE_GENERATION: "free_generation_label_parse",
    PREDICTION_MODE_ORIGINAL_TEACHER_FORCED: "teacher_forced_label_span",
}
SUPPORTED_PREDICTION_MODES = tuple(PREDICTION_MODE_PROTOCOLS.keys())
AGGREGATION_LEVEL_SUBJECT = "subject"
AGGREGATION_LEVEL_SEGMENT = "segment"
SUPPORTED_AGGREGATION_LEVELS = (
    AGGREGATION_LEVEL_SUBJECT,
    AGGREGATION_LEVEL_SEGMENT,
)
ENV_VAR_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(:-([^}]*))?\}")
PROJECT_PATH_ANCHORS = ("outputs", "output_model", "configs", "scripts", "src")
OPTIONAL_OVERRIDE_PATHS = {
    ("split", "cv_protocol"),
    ("split", "mode"),
    ("split", "dev_pool_partitions"),
    ("split", "outer_folds"),
    ("split", "smoke_subject_limit"),
}
GENERATION_PARSE_PREFIXES = (
    "answer:",
    "answer is",
    "label:",
    "label is",
    "prediction:",
    "prediction is",
    "predicted label:",
    "the label is",
    "the subject is",
    "subject is",
    "the speaker is",
    "speaker is",
    "it is",
    "it's",
    "classified as",
    "classification:",
)


def project_root() -> Path:
    env_root = os.environ.get("PROJECT_ROOT")
    if env_root:
        return Path(env_root).resolve()
    return Path(__file__).resolve().parents[1]


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _expand_string_value(value: str) -> str:
    def replace(match: re.Match[str]) -> str:
        var_name = match.group(1)
        default_value = match.group(3)
        if var_name == "PROJECT_ROOT":
            return os.environ.get("PROJECT_ROOT", str(project_root()))
        env_value = os.environ.get(var_name)
        if env_value not in (None, ""):
            return env_value
        if default_value is not None:
            return default_value
        return match.group(0)

    expanded = ENV_VAR_PATTERN.sub(replace, value)
    return os.path.expandvars(expanded)


def expand_env_vars(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: expand_env_vars(item) for key, item in value.items()}
    if isinstance(value, list):
        return [expand_env_vars(item) for item in value]
    if isinstance(value, str):
        return _expand_string_value(value)
    return value


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return expand_env_vars(data)


def _default_label_config_for_version(version: str) -> dict[str, str]:
    if version == LABEL_VOCAB_VERSION_SHORT_AB:
        return {
            "label_vocab_version": LABEL_VOCAB_VERSION_SHORT_AB,
            "internal_positive_label": "A",
            "internal_negative_label": "B",
            "external_positive_label": LABEL_DEPRESSED,
            "external_negative_label": LABEL_NON_DEPRESSED,
        }
    if version == LABEL_VOCAB_VERSION_LEGACY:
        return {
            "label_vocab_version": LABEL_VOCAB_VERSION_LEGACY,
            "internal_positive_label": LABEL_DEPRESSED,
            "internal_negative_label": LABEL_NON_DEPRESSED,
            "external_positive_label": LABEL_DEPRESSED,
            "external_negative_label": LABEL_NON_DEPRESSED,
        }
    raise ValueError(
        f"Unsupported labels.label_vocab_version={version!r}. "
        f"Expected one of {', '.join(SUPPORTED_LABEL_VOCAB_VERSIONS)}."
    )


def resolve_label_config(config: dict[str, Any]) -> dict[str, str]:
    labels_cfg = dict(config.get("labels", {}))
    version = str(labels_cfg.get("label_vocab_version", LABEL_VOCAB_VERSION_SHORT_AB)).strip()
    defaults = _default_label_config_for_version(version)
    resolved = dict(defaults)
    # Apply explicit customizations only when they do not simply mirror the
    # stale legacy YAML values from another vocabulary version.
    for key, value in labels_cfg.items():
        if key == "label_vocab_version":
            resolved[key] = value
            continue
        if key not in defaults:
            resolved[key] = value
            continue
        other_defaults = [
            cfg[key]
            for other_version in SUPPORTED_LABEL_VOCAB_VERSIONS
            if other_version != version
            for cfg in [_default_label_config_for_version(other_version)]
        ]
        if value == defaults[key]:
            resolved[key] = value
            continue
        if value in other_defaults:
            continue
        resolved[key] = value
    required_keys = (
        "internal_positive_label",
        "internal_negative_label",
        "external_positive_label",
        "external_negative_label",
        "label_vocab_version",
    )
    for key in required_keys:
        value = str(resolved.get(key, "")).strip()
        if not value:
            raise ValueError(f"Missing labels.{key} in config.")
        resolved[key] = value
    return resolved


def resolve_input_modality(config: dict[str, Any]) -> str:
    data_cfg = config.get("data", {})
    use_audio = bool(data_cfg.get("use_audio", False))
    use_text = bool(data_cfg.get("use_text", False))
    if use_audio and use_text:
        return INPUT_MODALITY_AUDIO_TEXT
    if use_audio:
        return INPUT_MODALITY_AUDIO_ONLY
    if use_text:
        return INPUT_MODALITY_TEXT_ONLY
    raise ValueError(
        "Invalid data.use_audio/data.use_text configuration. "
        "At least one modality must be enabled."
    )


def _parse_override_value(raw_value: str) -> Any:
    try:
        return yaml.safe_load(raw_value)
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid override value {raw_value!r}: {exc}") from exc


def parse_config_override(override: str) -> tuple[list[str], Any]:
    if "=" not in override:
        raise ValueError(f"Invalid override {override!r}. Expected KEY=VALUE.")
    raw_path, raw_value = override.split("=", 1)
    path = [part.strip() for part in raw_path.split(".") if part.strip()]
    if not path:
        raise ValueError(f"Invalid override {override!r}. Expected non-empty dotted KEY path.")
    return path, _parse_override_value(raw_value)


def normalize_config_overrides(overrides: list[str] | None) -> list[str]:
    normalized: list[str] = []
    if not overrides:
        return normalized

    expecting_value = False
    for override in overrides:
        token = str(override).strip()
        if not token:
            continue
        if expecting_value:
            normalized.append(token)
            expecting_value = False
            continue
        if token == "--set":
            expecting_value = True
            continue
        if token.startswith("--set="):
            token = token[len("--set=") :].strip()
            if not token:
                raise ValueError("Invalid override '--set='. Expected KEY=VALUE.")
        normalized.append(token)

    if expecting_value:
        raise ValueError("Invalid override '--set'. Expected KEY=VALUE after --set.")

    return normalized


def apply_config_overrides(config: dict[str, Any], overrides: list[str] | None) -> dict[str, Any]:
    resolved = expand_env_vars(dict(config))
    for override in normalize_config_overrides(overrides):
        path, value = parse_config_override(override)
        cursor: Any = resolved
        for part in path[:-1]:
            if not isinstance(cursor, dict) or part not in cursor:
                raise ValueError(f"Unknown config override path: {'.'.join(path)}")
            cursor = cursor[part]
        leaf_key = path[-1]
        if not isinstance(cursor, dict):
            raise ValueError(f"Unknown config override path: {'.'.join(path)}")
        if leaf_key not in cursor and tuple(path) not in OPTIONAL_OVERRIDE_PATHS:
            raise ValueError(f"Unknown config override path: {'.'.join(path)}")
        cursor[leaf_key] = expand_env_vars(value)
    return resolved


def load_yaml_with_overrides(path: str | Path, overrides: list[str] | None = None) -> dict[str, Any]:
    return apply_config_overrides(load_yaml(path), overrides)


def resolve_project_path(path: str | Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        return (project_root() / candidate).resolve()
    if candidate.exists():
        return candidate
    parts = candidate.parts
    for anchor in PROJECT_PATH_ANCHORS:
        if anchor in parts:
            anchor_index = parts.index(anchor)
            relocated = project_root().joinpath(*parts[anchor_index:])
            return relocated
    return candidate


def serialize_project_path(path: str | Path) -> str:
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(project_root()))
    except ValueError:
        return str(resolved)


def resolve_metadata_paths(metadata: dict[str, Any]) -> dict[str, Any]:
    resolved = dict(metadata)
    for key, value in list(resolved.items()):
        if key.endswith("_path") and isinstance(value, str) and value:
            resolved[key] = str(resolve_project_path(value))
    return resolved


def save_yaml(data: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False, allow_unicode=True)


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(data: Any, path: str | Path, indent: int = 2) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=indent, ensure_ascii=False)
        handle.write("\n")


def save_json_atomic(data: Any, path: str | Path, indent: int = 2) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    tmp_path = path.with_name(f"{path.name}.tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=indent, ensure_ascii=False)
        handle.write("\n")
    tmp_path.replace(path)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(rows: list[dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_jsonl_rows(rows: list[dict[str, Any]]) -> str:
    serialized = "\n".join(json.dumps(row, sort_keys=True, ensure_ascii=False) for row in rows)
    return sha256_text(serialized)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def configure_logging(verbose: bool = True) -> None:
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(message)s")


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def label_text_from_int(label: int) -> str:
    if label not in LABEL_TEXT_BY_INT:
        raise ValueError(f"Unexpected label value: {label}")
    return LABEL_TEXT_BY_INT[label]


def label_int_from_text(label_text: str) -> int:
    if label_text not in LABEL_INT_BY_TEXT:
        raise ValueError(f"Unexpected label text: {label_text}")
    return LABEL_INT_BY_TEXT[label_text]


def external_label_text_from_int(config: dict[str, Any], label: int) -> str:
    labels_cfg = resolve_label_config(config)
    if label == 1:
        return labels_cfg["external_positive_label"]
    if label == 0:
        return labels_cfg["external_negative_label"]
    raise ValueError(f"Unexpected label value: {label}")


def internal_label_text_from_int(config: dict[str, Any], label: int) -> str:
    labels_cfg = resolve_label_config(config)
    if label == 1:
        return labels_cfg["internal_positive_label"]
    if label == 0:
        return labels_cfg["internal_negative_label"]
    raise ValueError(f"Unexpected label value: {label}")


def _normalized_label_text(text: str) -> str:
    return " ".join(str(text).strip().split()).lower()


def _compact_label_text(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(text).lower())


def parse_internal_label_text(text: str, config: dict[str, Any]) -> int | None:
    normalized = _normalized_label_text(text)
    if not normalized:
        return None
    labels_cfg = resolve_label_config(config)
    positive = _normalized_label_text(labels_cfg["internal_positive_label"])
    negative = _normalized_label_text(labels_cfg["internal_negative_label"])
    if normalized == positive:
        return 1
    if normalized == negative:
        return 0
    return None


def _candidate_generation_fragments(text: str) -> list[str]:
    normalized = _normalized_label_text(text)
    if not normalized:
        return []

    candidates: list[str] = [normalized]
    for prefix in GENERATION_PARSE_PREFIXES:
        if normalized.startswith(prefix):
            stripped = normalized[len(prefix) :].strip()
            if stripped:
                candidates.append(stripped)

    words = re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)?", normalized)
    for width in range(1, min(6, len(words)) + 1):
        candidates.append(" ".join(words[-width:]))

    deduped: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        candidate = candidate.strip()
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        deduped.append(candidate)
    return deduped


def _prefix_match_label(candidate: str, label: str) -> bool:
    candidate_compact = _compact_label_text(candidate)
    label_compact = _compact_label_text(label)
    if not candidate_compact or not label_compact:
        return False
    if len(label_compact) <= 2:
        return candidate_compact == label_compact
    min_prefix_len = min(len(label_compact), max(4, len(label_compact) // 2))
    return len(candidate_compact) >= min_prefix_len and label_compact.startswith(candidate_compact)


def _first_regex_match(text: str, patterns: tuple[str, ...]) -> re.Match[str] | None:
    earliest: re.Match[str] | None = None
    for pattern in patterns:
        match = re.search(pattern, text)
        if match is None:
            continue
        if earliest is None or match.start() < earliest.start():
            earliest = match
    return earliest


def parse_generated_label_text(text: str, config: dict[str, Any]) -> int | None:
    labels_cfg = resolve_label_config(config)
    version = labels_cfg["label_vocab_version"]
    positive_label = labels_cfg["internal_positive_label"]
    negative_label = labels_cfg["internal_negative_label"]
    positive_norm = _normalized_label_text(positive_label)
    negative_norm = _normalized_label_text(negative_label)
    positive_compact = _compact_label_text(positive_label)
    negative_compact = _compact_label_text(negative_label)
    normalized_full = _normalized_label_text(text)

    if not normalized_full:
        return None

    if version == LABEL_VOCAB_VERSION_LEGACY:
        negative_patterns = (
            r"\bnon[\s-]*depressed\b",
            r"\bnondepressed\b",
            r"\bnot depressed\b",
            r"\bno depression\b",
            r"\bwithout depression\b",
        )
        positive_patterns = (
            r"\bdepressed\b",
            r"\bhas depression\b",
            r"\bwith depression\b",
        )

        negative_match = _first_regex_match(normalized_full, negative_patterns)
        positive_match = _first_regex_match(normalized_full, positive_patterns)
        if negative_match and (not positive_match or negative_match.start() <= positive_match.start()):
            return 0
        if positive_match:
            return 1

    candidates = _candidate_generation_fragments(text)
    for candidate in candidates:
        normalized = _normalized_label_text(candidate)
        compact = _compact_label_text(candidate)
        if normalized == positive_norm or compact == positive_compact:
            return 1
        if normalized == negative_norm or compact == negative_compact:
            return 0

    for candidate in candidates:
        pos_prefix = _prefix_match_label(candidate, positive_label)
        neg_prefix = _prefix_match_label(candidate, negative_label)
        if pos_prefix and not neg_prefix:
            return 1
        if neg_prefix and not pos_prefix:
            return 0

    return None


def prompt_label_descriptor(config: dict[str, Any]) -> str:
    labels_cfg = resolve_label_config(config)
    return (
        f"{labels_cfg['external_positive_label']} or "
        f"{labels_cfg['external_negative_label']}"
    )


def prompt_label_instruction(config: dict[str, Any]) -> str:
    labels_cfg = resolve_label_config(config)
    version = labels_cfg["label_vocab_version"]
    if version == LABEL_VOCAB_VERSION_SHORT_AB:
        return (
            f"Use this label legend:\n"
            f"{labels_cfg['internal_positive_label']} = {labels_cfg['external_positive_label']}\n"
            f"{labels_cfg['internal_negative_label']} = {labels_cfg['external_negative_label']}\n"
            f"Answer with exactly one label: "
            f"{labels_cfg['internal_positive_label']} or {labels_cfg['internal_negative_label']}."
        )
    return (
        "Answer with exactly one label: "
        f"{labels_cfg['internal_positive_label']} or {labels_cfg['internal_negative_label']}."
    )


def resolve_model_name_or_path(cli_value: str | None, config: dict[str, Any]) -> str:
    if cli_value:
        return cli_value
    env_value = os.environ.get("MODEL_PATH")
    if env_value:
        return env_value
    return str(config["model_name_or_path"])


def to_serializable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: to_serializable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_serializable(item) for item in value]
    return value


def print_json(data: Any) -> None:
    print(json.dumps(to_serializable(data), indent=2, ensure_ascii=False))


def format_config_overrides_for_logging(overrides: list[str] | None) -> str:
    if not overrides:
        return "<none>"
    return "\n".join(f"  - {item}" for item in overrides)


def format_resolved_config_for_logging(config: dict[str, Any]) -> str:
    return yaml.safe_dump(to_serializable(config), sort_keys=False, allow_unicode=True).rstrip()


def log_resolved_config(
    logger: logging.Logger,
    *,
    base_config_path: str | Path,
    config_overrides: list[str] | None,
    resolved_config: dict[str, Any],
) -> None:
    logger.info("========================================")
    logger.info("Base Config: %s", Path(base_config_path))
    logger.info("Config Overrides:\n%s", format_config_overrides_for_logging(config_overrides))
    logger.info("Resolved Config:\n%s", format_resolved_config_for_logging(resolved_config))
    logger.info("========================================")


def normalize_prediction_mode(value: str | None) -> str:
    normalized = (value or PREDICTION_MODE_LIKELIHOOD).strip().lower()
    if normalized not in SUPPORTED_PREDICTION_MODES:
        raise ValueError(
            f"Unsupported evaluation.sample_prediction_mode={value!r}. "
            f"Expected one of {', '.join(SUPPORTED_PREDICTION_MODES)}."
        )
    return normalized


def resolve_prediction_mode(config: dict[str, Any], override: str | None = None) -> str:
    if override is not None:
        return normalize_prediction_mode(override)
    evaluation_cfg = config.get("evaluation", {})
    return normalize_prediction_mode(evaluation_cfg.get("sample_prediction_mode"))


def normalize_aggregation_level(value: str | None) -> str:
    normalized = (value or AGGREGATION_LEVEL_SUBJECT).strip().lower()
    if normalized not in SUPPORTED_AGGREGATION_LEVELS:
        raise ValueError(
            f"Unsupported evaluation.aggregation_level={value!r}. "
            f"Expected one of {', '.join(SUPPORTED_AGGREGATION_LEVELS)}."
        )
    return normalized


def resolve_aggregation_level(config: dict[str, Any], override: str | None = None) -> str:
    if override is not None:
        return normalize_aggregation_level(override)
    evaluation_cfg = config.get("evaluation", {})
    return normalize_aggregation_level(evaluation_cfg.get("aggregation_level"))


def evaluation_protocol_name(mode: str) -> str:
    normalized = normalize_prediction_mode(mode)
    return PREDICTION_MODE_PROTOCOLS[normalized]
