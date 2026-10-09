"""Focused regressions for extraction-source compatibility cache reuse.

The audited compatibility record may authorize reuse of a complete hidden cache
only when the recorded and current identities differ exclusively in
``source_git_commit`` and the record proves the full extraction-relevant file
closure byte-identical apart from the reviewed compatibility-only extractor
delta. Anything unknown or missing must fail closed.
"""
from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from src.features import extract_qwen_hidden as ex

DELTA = "src/features/extract_qwen_hidden.py"
OTHER = "src/utils.py"
OLD_COMMIT = "a" * 40
NEW_COMMIT = "b" * 40


def _record(
    *,
    old_commit: str = OLD_COMMIT,
    delta_old: str = "1" * 64,
    delta_new: str = "2" * 64,
    other: str = "3" * 64,
    old_other: str | None = None,
    new_delta: str | None = None,
    include_other_old: bool = True,
) -> dict:
    old_files = {DELTA: delta_old}
    if include_other_old:
        old_files[OTHER] = old_other if old_other is not None else other
    return {
        "schema": ex.EXTRACTION_SOURCE_COMPAT_SCHEMA,
        "old_sources": [
            {
                "git_commit": old_commit,
                "deployment_id": "old-deployment",
                "file_sha256": old_files,
            }
        ],
        "extractor_delta": {"path": DELTA, "old_sha256": delta_old, "new_sha256": delta_new},
        "new_source_file_sha256": {
            DELTA: new_delta if new_delta is not None else delta_new,
            OTHER: other,
        },
    }


class ExtractionSourceCompatTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.tmp_path = Path(self.tmp.name)
        self.existing = {"source_git_commit": OLD_COMMIT, "dataset": "androids_interview"}
        self.current = {"source_git_commit": NEW_COMMIT, "dataset": "androids_interview"}

    def _allow(
        self,
        record: dict,
        *,
        existing: dict | None = None,
        current: dict | None = None,
        files: dict[str, str] | None = None,
        record_path: Path | None = None,
    ) -> bool:
        path = record_path if record_path is not None else self.tmp_path / "record.json"
        if record_path is None:
            path.write_text(json.dumps(record), encoding="utf-8")
        return ex._extraction_source_compat_reuse_allowed(
            existing if existing is not None else self.existing,
            current if current is not None else self.current,
            record_path=path,
            current_file_sha256=files if files is not None else {DELTA: "2" * 64, OTHER: "3" * 64},
        )

    def test_valid_record_allows_reuse(self) -> None:
        self.assertTrue(self._allow(_record()))

    def test_any_other_identity_difference_refuses(self) -> None:
        current = dict(self.current)
        current["dataset"] = "d3tec"
        self.assertFalse(self._allow(_record(), current=current))

    def test_missing_source_commit_key_refuses(self) -> None:
        self.assertFalse(self._allow(_record(), existing={"dataset": "androids_interview"}))

    def test_unknown_old_commit_refuses(self) -> None:
        self.assertFalse(
            self._allow(
                _record(),
                existing={"source_git_commit": "c" * 40, "dataset": "androids_interview"},
            )
        )

    def test_missing_record_refuses(self) -> None:
        self.assertFalse(self._allow(_record(), record_path=self.tmp_path / "absent.json"))

    def test_altered_current_extractor_refuses(self) -> None:
        self.assertFalse(self._allow(_record(), files={DELTA: "9" * 64, OTHER: "3" * 64}))

    def test_altered_current_dependency_refuses(self) -> None:
        self.assertFalse(self._allow(_record(), files={DELTA: "2" * 64, OTHER: "9" * 64}))

    def test_unequal_non_delta_file_refuses(self) -> None:
        self.assertFalse(self._allow(_record(old_other="4" * 64)))

    def test_old_file_set_mismatch_refuses(self) -> None:
        self.assertFalse(self._allow(_record(include_other_old=False)))

    def test_delta_new_hash_mismatch_refuses(self) -> None:
        self.assertFalse(self._allow(_record(new_delta="5" * 64)))

    def test_equal_delta_hashes_refuse(self) -> None:
        self.assertFalse(
            self._allow(_record(delta_old="6" * 64, delta_new="6" * 64, new_delta="6" * 64))
        )

    def test_exact_identity_still_skips_without_record(self) -> None:
        output = self.tmp_path / "exact"
        output.mkdir()
        config = {"source_git_commit": OLD_COMMIT, "dataset": "androids_interview"}
        (output / "extraction_metadata.json").write_text(
            json.dumps({"cache_config": config, "cache_config_sha256": "s"}), encoding="utf-8"
        )
        for name in ex.CACHE_ARTIFACT_NAMES:
            if name != "extraction_metadata.json":
                (output / name).write_text("x", encoding="utf-8")
        self.assertEqual(
            ex._existing_cache_decision(output, config, "s"),
            "skipped_compatible_complete_cache",
        )

    def _real_hashes(self) -> tuple[str, str]:
        repo = Path(ex.__file__).resolve().parents[2]
        return ex.sha256_file(repo / DELTA), ex.sha256_file(repo / OTHER)

    def _complete_cache(self, output: Path, existing_config: dict, cache_config: dict) -> None:
        output.mkdir()
        (output / "extraction_metadata.json").write_text(
            json.dumps(
                {"cache_config": existing_config, "cache_config_sha256": ex.sha256_text("x")}
            ),
            encoding="utf-8",
        )
        for name in ex.CACHE_ARTIFACT_NAMES:
            if name != "extraction_metadata.json":
                (output / name).write_text("x", encoding="utf-8")

    def test_decision_uses_record_for_complete_cache(self) -> None:
        delta_sha, other_sha = self._real_hashes()
        record = _record(delta_old="7" * 64, delta_new=delta_sha, other=other_sha)
        record_path = self.tmp_path / "record.json"
        record_path.write_text(json.dumps(record), encoding="utf-8")
        original = ex.EXTRACTION_SOURCE_COMPAT_PATH
        ex.EXTRACTION_SOURCE_COMPAT_PATH = record_path
        try:
            output = self.tmp_path / "cache"
            self._complete_cache(output, self.existing, self.current)
            self.assertEqual(
                ex._existing_cache_decision(output, self.current, ex.sha256_text("y")),
                ex.CACHE_DECISION_COMPAT_REUSE,
            )
            # A tampered running dependency must refuse even with the record present.
            real_sha256_file = ex.sha256_file
            ex.sha256_file = lambda path: (
                "0" * 64 if str(path).endswith("utils.py") else real_sha256_file(path)
            )
            try:
                with self.assertRaisesRegex(ValueError, "incompatible"):
                    ex._existing_cache_decision(output, self.current, ex.sha256_text("y"))
            finally:
                ex.sha256_file = real_sha256_file
        finally:
            ex.EXTRACTION_SOURCE_COMPAT_PATH = original

    def test_partial_compat_cache_still_refused(self) -> None:
        delta_sha, other_sha = self._real_hashes()
        record = _record(delta_new=delta_sha, other=other_sha)
        record_path = self.tmp_path / "record.json"
        record_path.write_text(json.dumps(record), encoding="utf-8")
        original = ex.EXTRACTION_SOURCE_COMPAT_PATH
        ex.EXTRACTION_SOURCE_COMPAT_PATH = record_path
        try:
            output = self.tmp_path / "partial"
            output.mkdir()
            (output / "extraction_metadata.json").write_text(
                json.dumps(
                    {"cache_config": self.existing, "cache_config_sha256": ex.sha256_text("x")}
                ),
                encoding="utf-8",
            )
            (output / "outer_train.npz").write_text("x", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "partial"):
                ex._existing_cache_decision(output, self.current, ex.sha256_text("y"))
        finally:
            ex.EXTRACTION_SOURCE_COMPAT_PATH = original

    def test_repository_record_matches_running_source(self) -> None:
        """The tracked record must match the running closure exactly (fail closed otherwise)."""
        self.assertTrue(ex.EXTRACTION_SOURCE_COMPAT_PATH.is_file())
        record = json.loads(ex.EXTRACTION_SOURCE_COMPAT_PATH.read_text(encoding="utf-8"))
        repo = Path(ex.__file__).resolve().parents[2]
        for relative_path, expected_sha in record["new_source_file_sha256"].items():
            self.assertEqual(ex.sha256_file(repo / relative_path), expected_sha, relative_path)


if __name__ == "__main__":
    unittest.main()
