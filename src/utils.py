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
ENV_VAR_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(:-([^}]*))?\}")
PROJECT_PATH_ANCHORS = ("outputs", "output_model", "configs", "scripts", "src")


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
