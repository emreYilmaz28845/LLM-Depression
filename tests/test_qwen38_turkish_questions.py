"""Turkish question-recovery pipeline tests: prepare, resume, consolidation,
aggregation rules, deterministic rendering, and privacy checks."""
from __future__ import annotations

import hashlib
import json

import pytest

from src.qwen38.audit import (
    _compact_texts,
    _recompute_restricted_evidence,
    ENV_PINS,
    audit_turkish,
    ngram_overlap_at_least,
)
from src.qwen38.contracts import (
    FINAL_TABLE_COLUMNS,
    MODEL_ID,
    MODEL_REVISION,
    TURKISH_MAX_TOKENS,
    TURKISH_SOURCE_HASH,
    WordingStatus,
    generation_settings_hash,
    parse_filename_stem,
    request_settings,
)
from src.qwen38.turkish_questions import (
    EVIDENCE_BASIS_FALLBACK,
    PROMPT_VERSION,
    _check_cluster_assignment,
    _check_family_assignment,
    _episode_provenance,
    _validate_episodes,
    aggregate_families,
    collect_candidates,
    load_prepared_sequences,
    load_table_rows,
    prepare_sequences,
    prompt_bundle_sha256,
    prompt_contract_sha256,
    prompt_component_hashes,
    render_tables,
    sanitize_evidence_basis,
)

FIXTURE = "tests/fixtures/qwen38_synthetic_cases.jsonl"


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_file_for_test(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_transcript(path, subjects_windows, transcripts=None):
    """Write a synthetic transcript JSONL; returns its sha256."""
    rows = []
    index = 0
    for subject, windows in subjects_windows.items():
        for window in windows:
            condition = ["ank", "depr", "depr+ank", "ank+depr", "dep+ank"][index % 5]
            stem = f"{subject}-1-{window}-{condition}"
            text = transcripts[subject][window] if transcripts else f"cevap {subject} {window}"
            rows.append(
                {
                    "audio_path": f"/media/emre/Backup/AudioLLM/Datasets/Turkish/all-files/{stem}.wav",
                    "transcript": text,
                    "asr_model": "Qwen/Qwen3-ASR-1.7B",
                    "language": "tr",
                    "repair_status": "QWEN3ASR_RAW",
                }
            )
            index += 1
    payload = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n"
    path.write_text(payload, encoding="utf-8")
    return _sha256_text(payload)


def _make_inference_record(sequence_id, episodes, prompt_hash, source_sha256, source_commit, model_revision="1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0", sequence=None):
    record = {
        "sequence_id": sequence_id,
        "status": "completed",
        "prompt_hash": prompt_hash,
        "source_sha256": source_sha256,
        "source_commit": source_commit,
        "model_revision": model_revision,
        "generation_settings_hash": generation_settings_hash(2048),
        "episode_count": len(episodes),
        "episodes": episodes,
    }
    if sequence is not None:
        for key in (
            "turkish_run_id",
            "analysis_attempt",
            "deployment_id",
            "model_id",
            "prompt_version",
            "run_manifest_sha256",
            "user_prompt_sha256",
            "system_prompt_sha256",
            "correction_message_sha256",
            "subject_schema_sha256",
            "prompt_contract_sha256",
            "prompt_bundle_sha256",
        ):
            record[key] = sequence[key]
        record["prompt_hash"] = sequence["prompt_hash"]
    return record


def _episode(sequence_id, order, label, confidence, wording=WordingStatus.INFERRED_PARAPHRASE.value, abstain=""):
    return {
        "sequence_id": sequence_id,
        "episode_order": order,
        "question_tr": f"tr soru {sequence_id} {order}",
        "question_en": f"en question {sequence_id} {order}",
        "label": label,
        "wording_status": wording,
        "confidence": confidence,
        "evidence_window_indices": [order],
        "evidence_basis": f"kisa aciklama {order}",
        "abstain_reason": abstain,
    }


class TestPrepare:
    def test_grouping_by_subject_sequence(self, tmp_path):
        transcript = tmp_path / "t.jsonl"
        source_hash = _write_transcript(
            transcript,
            {"subA": [2, 1, 3], "subB": [1], "subC": [4, 5]},
        )
        run_dir = tmp_path / "run"
        summary = prepare_sequences(
            transcript,
            run_dir=run_dir,
            deployment_id="qwen38_test",
            model_revision="r",
            source_commit="c",
            source_sha256=source_hash,
            expected_sequences=3,
            expected_windows=6,
        )
        assert summary["sequences"] == 3
        assert summary["windows"] == 6
        sequences = load_prepared_sequences(run_dir / "restricted" / "prepared_sequences.jsonl")
        subject_map = json.loads(
            (run_dir / "restricted" / "subject_map.json").read_text(encoding="utf-8")
        )
        sequence_for = {subject: seq_id for seq_id, subject in subject_map.items()}
        sub_a_id = sequence_for["subA"]
        by_id = {seq["sequence_id"]: seq for seq in sequences}
        assert len(by_id) == 3
        # Windows are ordered numerically, not by file order.
        assert by_id[sub_a_id]["windows"] == [
            {"window": 1, "text": "cevap subA 1"},
            {"window": 2, "text": "cevap subA 2"},
            {"window": 3, "text": "cevap subA 3"},
        ]
        assert "[WINDOW 1]" in by_id[sub_a_id]["user_prompt"]
        assert "[WINDOW 2]" in by_id[sub_a_id]["user_prompt"]
        # Grouping is per subject sequence; a subject's windows never mix.
        for seq in sequences:
            for window in seq["windows"]:
                assert window["window"] in (1, 2, 3, 4, 5)

    def test_sequence_ids_by_subject_hash(self, tmp_path):
        transcript = tmp_path / "t.jsonl"
        source_hash = _write_transcript(
            transcript, {"zzz": [1], "aaa": [1], "mmm": [1]}
        )
        run_dir = tmp_path / "run"
        prepare_sequences(
            transcript,
            run_dir=run_dir,
            deployment_id="d",
            model_revision="r",
            source_commit="c",
            source_sha256=source_hash,
            expected_sequences=3,
            expected_windows=3,
        )
        subject_map = json.loads(
            (run_dir / "restricted" / "subject_map.json").read_text(encoding="utf-8")
        )
        ordered = sorted(subject_map, key=lambda seq: subject_map[seq])
        assert len(subject_map) == 3
        hashes = [_sha256_text(subject_map[seq]) for seq in sorted(subject_map)]
        assert hashes == sorted(hashes)

    def test_no_condition_tag_in_packets(self, tmp_path):
        transcript = tmp_path / "t.jsonl"
        source_hash = _write_transcript(
            transcript, {"subA": [1, 2, 3], "subB": [1, 2]}, transcripts={
                "subA": {1: "a1", 2: "a2", 3: "a3"},
                "subB": {1: "b1", 2: "b2"},
            }
        )
        run_dir = tmp_path / "run"
        prepare_sequences(
            transcript,
            run_dir=run_dir,
            deployment_id="d",
            model_revision="r",
            source_commit="c",
            source_sha256=source_hash,
            expected_sequences=2,
            expected_windows=5,
        )
        for sequence in load_prepared_sequences(run_dir / "restricted" / "prepared_sequences.jsonl"):
            assert "condition" not in sequence
            for window in sequence["windows"]:
                assert "condition" not in window
            for tag in ("ank", "depr"):
                assert tag not in sequence["user_prompt"]

    def test_source_hash_enforced(self, tmp_path):
        transcript = tmp_path / "t.jsonl"
        source_hash = _write_transcript(transcript, {"subA": [1, 2]})
        with pytest.raises(ValueError, match="hash mismatch"):
            prepare_sequences(
                transcript,
                run_dir=tmp_path / "run",
                deployment_id="d",
                model_revision="r",
                source_commit="c",
                source_sha256="0" * 64,
                expected_sequences=1,
                expected_windows=2,
            )

    def test_duplicate_window_pair_rejected(self, tmp_path):
        transcript = tmp_path / "t.jsonl"
        source_hash = _write_transcript(transcript, {"subA": [1, 1]})
        with pytest.raises(ValueError, match="duplicate"):
            prepare_sequences(
                transcript,
                run_dir=tmp_path / "run",
                deployment_id="d",
                model_revision="r",
                source_commit="c",
                source_sha256=source_hash,
                expected_sequences=1,
                expected_windows=2,
            )

    def test_empty_transcript_rejected(self, tmp_path):
        transcript = tmp_path / "t.jsonl"
        payload = json.dumps(
            {
                "audio_path": "/x/subA-1-1-ank.wav",
                "transcript": "   ",
            },
            ensure_ascii=False,
        )
        transcript.write_text(payload + "\n", encoding="utf-8")
        with pytest.raises(ValueError, match="empty transcript"):
            prepare_sequences(
                transcript,
                run_dir=tmp_path / "run",
                deployment_id="d",
                model_revision="r",
                source_commit="c",
                source_sha256=_sha256_text(payload + "\n"),
                expected_sequences=1,
                expected_windows=1,
            )

    def test_run_root_collision_refused(self, tmp_path):
        transcript = tmp_path / "t.jsonl"
        source_hash = _write_transcript(transcript, {"subA": [1]})
        run_dir = tmp_path / "run"
        prepare_sequences(
            transcript,
            run_dir=run_dir,
            deployment_id="d",
            model_revision="r",
            source_commit="c",
            source_sha256=source_hash,
            expected_sequences=1,
            expected_windows=1,
        )
        with pytest.raises(ValueError, match="reuse existing"):
            prepare_sequences(
                transcript,
                run_dir=run_dir,
                deployment_id="d",
                model_revision="r",
                source_commit="c",
                source_sha256=source_hash,
                expected_sequences=1,
                expected_windows=1,
            )


class TestPromptIdentityAndSanitization:
    def test_prompt_version_is_v2(self):
        from src.qwen38.turkish_questions import PROMPT_VERSION

        assert PROMPT_VERSION == "qwen38_turkish_v2"

    def test_prompt_contract_and_bundle_are_deterministic(self):
        contract_a = prompt_contract_sha256(model_revision="r")
        contract_b = prompt_contract_sha256(model_revision="r")
        assert contract_a == contract_b
        assert prompt_bundle_sha256("same", model_revision="r") == prompt_bundle_sha256("same", model_revision="r")
        assert prompt_bundle_sha256("different", model_revision="r") != prompt_bundle_sha256("same", model_revision="r")

    def test_sanitizer_removes_all_quotes_and_normalizes(self):
        value = '  A" B\' C` D“ E” F‘ G’ H« I» J‹ K› L„ M‟ N‚ O‛  '
        sanitized = sanitize_evidence_basis(value)
        assert sanitized == "A B C D E F G H I J K L M N O"

    def test_sanitizer_uses_word_boundary_and_fallback(self):
        long_value = "kelime " * 80
        sanitized = sanitize_evidence_basis(long_value)
        assert len(sanitized) <= 200
        assert not sanitized.endswith("kel")
        assert sanitize_evidence_basis("\n\t\r  ") == EVIDENCE_BASIS_FALLBACK

    def test_sanitized_evidence_still_receives_privacy_check(self):
        payload = {
            "episodes": [{
                "sequence_id": "S0001",
                "episode_order": 1,
                "question_tr": "Soru nedir?",
                "question_en": "What is the question?",
                "label": "NEUTRAL",
                "wording_status": "INFERRED_PARAPHRASE",
                "confidence": "LOW",
                "evidence_window_indices": [1],
                "evidence_basis": 'Bugün sabah erkenden pazara gittim ve taze meyve aldım sonra eve döndüm',
                "abstain_reason": "",
            }]
        }
        with pytest.raises(ValueError, match="12-token transcript overlap"):
            _validate_episodes(
                payload,
                "S0001",
                ["Bugün sabah erkenden pazara gittim ve taze meyve aldım sonra eve döndüm"],
            )


class TestInferenceResume:
    def _prepared(self, tmp_path):
        transcript = tmp_path / "t.jsonl"
        source_hash = _write_transcript(transcript, {"subA": [1, 2], "subB": [1]})
        run_dir = tmp_path / "run"
        prepare_sequences(
            transcript,
            run_dir=run_dir,
            deployment_id="d",
            model_revision="r",
            source_commit="c",
            source_sha256=source_hash,
            expected_sequences=2,
            expected_windows=3,
        )
        return run_dir, source_hash

    def test_resume_skips_completed_with_matching_provenance(self, tmp_path):
        from src.qwen38.turkish_questions import infer_subjects

        run_dir, source_hash = self._prepared(tmp_path)
        inferences = run_dir / "restricted" / "subject_inferences"
        inferences.mkdir(exist_ok=True)
        sequences = load_prepared_sequences(run_dir / "restricted" / "prepared_sequences.jsonl")
        for sequence in sequences:
            record = _make_inference_record(
                sequence["sequence_id"],
                [_episode(sequence["sequence_id"], 1, "NEUTRAL", "LOW")],
                prompt_hash=sequence["prompt_hash"],
                source_sha256=sequence["source_sha256"],
                source_commit=sequence["source_commit"],
                model_revision=sequence["model_revision"],
                sequence=sequence,
            )
            (inferences / f"{sequence['sequence_id']}.json").write_text(
                json.dumps(record, ensure_ascii=False), encoding="utf-8"
            )
        # No server is touched: everything is already completed.
        summary = infer_subjects(
            run_dir / "restricted" / "prepared_sequences.jsonl",
            inferences,
            base_url="http://127.0.0.1:1/v1",
        )
        assert summary["complete"]
        assert summary["completed_before_resume"] == 2
        assert summary["completed_now"] == 0

    def test_resume_refuses_provenance_change(self, tmp_path):
        from src.qwen38.turkish_questions import infer_subjects

        run_dir, source_hash = self._prepared(tmp_path)
        inferences = run_dir / "restricted" / "subject_inferences"
        inferences.mkdir(exist_ok=True)
        sequences = load_prepared_sequences(run_dir / "restricted" / "prepared_sequences.jsonl")
        record = _make_inference_record(
            sequences[0]["sequence_id"],
            [_episode(sequences[0]["sequence_id"], 1, "NEUTRAL", "LOW")],
            prompt_hash="changed-hash",
            source_sha256=sequences[0]["source_sha256"],
            source_commit=sequences[0]["source_commit"],
        )
        (inferences / f"{sequences[0]['sequence_id']}.json").write_text(
            json.dumps(record, ensure_ascii=False), encoding="utf-8"
        )
        with pytest.raises(ValueError, match="resume refused"):
            infer_subjects(
                run_dir / "restricted" / "prepared_sequences.jsonl",
                inferences,
                base_url="http://127.0.0.1:1/v1",
            )

    def test_changed_system_prompt_refuses_resume(self, tmp_path, monkeypatch):
        from src.qwen38.turkish_questions import infer_subjects

        run_dir, _ = self._prepared(tmp_path)
        monkeypatch.setattr(
            "src.qwen38.turkish_questions.SUBJECT_SYSTEM_PROMPT",
            "changed system prompt",
        )
        with pytest.raises(ValueError, match="resume refused"):
            infer_subjects(
                run_dir / "restricted" / "prepared_sequences.jsonl",
                run_dir / "restricted" / "subject_inferences",
                base_url="http://127.0.0.1:1/v1",
            )

    def test_changed_source_commit_refuses_resume(self, tmp_path):
        from src.qwen38.turkish_questions import infer_subjects

        run_dir, _ = self._prepared(tmp_path)
        with pytest.raises(ValueError, match="source commit changed"):
            infer_subjects(
                run_dir / "restricted" / "prepared_sequences.jsonl",
                run_dir / "restricted" / "subject_inferences",
                base_url="http://127.0.0.1:1/v1",
                source_commit="changed-source",
            )


class TestConsolidation:
    def test_cluster_assignment_exact_once(self):
        assignment = _check_cluster_assignment(
            [
                {
                    "cluster_id": "b1-c1",
                    "canonical_question_tr": "q",
                    "canonical_question_en": "q",
                    "member_candidate_ids": ["S0001-e1", "S0002-e1"],
                },
                {
                    "cluster_id": "b1-c2",
                    "canonical_question_tr": "q",
                    "canonical_question_en": "q",
                    "member_candidate_ids": ["S0003-e1"],
                },
            ],
            ["S0001-e1", "S0002-e1", "S0003-e1"],
            "test",
        )
        assert len(assignment) == 3

    def test_cluster_assignment_rejects_duplicate(self):
        with pytest.raises(ValueError, match="assigned twice"):
            _check_cluster_assignment(
                [
                    {
                        "cluster_id": "b1-c1",
                        "canonical_question_tr": "q",
                        "canonical_question_en": "q",
                        "member_candidate_ids": ["S0001-e1"],
                    },
                    {
                        "cluster_id": "b1-c2",
                        "canonical_question_tr": "q",
                        "canonical_question_en": "q",
                        "member_candidate_ids": ["S0001-e1"],
                    },
                ],
                ["S0001-e1"],
                "test",
            )

    def test_cluster_assignment_rejects_unassigned(self):
        with pytest.raises(ValueError, match="unassigned"):
            _check_cluster_assignment(
                [
                    {
                        "cluster_id": "b1-c1",
                        "canonical_question_tr": "q",
                        "canonical_question_en": "q",
                        "member_candidate_ids": ["S0001-e1"],
                    }
                ],
                ["S0001-e1", "S0002-e1"],
                "test",
            )

    def test_family_assignment_exact_once(self):
        assignment = _check_family_assignment(
            [
                {
                    "family_id": "f1",
                    "question_tr": "q",
                    "question_en": "q",
                    "member_cluster_ids": ["b1-c1", "b2-c3"],
                }
            ],
            ["b1-c1", "b2-c3"],
        )
        assert len(assignment) == 2

    def test_family_assignment_rejects_missing_cluster(self):
        with pytest.raises(ValueError, match="unassigned"):
            _check_family_assignment(
                [
                    {
                        "family_id": "f1",
                        "question_tr": "q",
                        "question_en": "q",
                        "member_cluster_ids": ["b1-c1"],
                    }
                ],
                ["b1-c1", "b2-c3"],
            )


def _candidate_rows(records):
    return collect_candidates(records)


def _family_setup():
    """Two sequences, one family with three candidates."""
    prompt_hash = "p"
    source_hash = "s"
    commit = "c"
    records = [
        _make_inference_record(
            "S0001",
            [
                _episode("S0001", 1, "POSITIVE", "HIGH", wording=WordingStatus.EXPLICIT_ECHO.value),
                _episode("S0001", 2, "POSITIVE", "HIGH"),
            ],
            prompt_hash,
            source_hash,
            commit,
        ),
        _make_inference_record(
            "S0002",
            [
                _episode("S0002", 1, "POSITIVE", "MEDIUM", wording=WordingStatus.EXPLICIT_ECHO.value),
            ],
            prompt_hash,
            source_hash,
            commit,
        ),
    ]
    candidates = _candidate_rows(records)
    clusters = [
        {
            "cluster_id": "b1-c1",
            "canonical_question_tr": "tr",
            "canonical_question_en": "en",
            "member_candidate_ids": [c["candidate_id"] for c in candidates],
        }
    ]
    cluster_to_candidate = {clusters[0]["cluster_id"]: clusters[0]["member_candidate_ids"]}
    families = [
        {
            "family_id": "f1",
            "question_tr": "Sizi mutlu eden şeyler nelerdir?",
            "question_en": "What makes you happy?",
            "member_cluster_ids": ["b1-c1"],
        }
    ]
    cluster_assignment = {"b1-c1": "f1"}
    return records, candidates, families, cluster_to_candidate, cluster_assignment


class TestAggregation:
    def test_positive_winner_accepted(self):
        records, candidates, families, cluster_to_candidate, cluster_assignment = _family_setup()
        rows = aggregate_families(candidates, records, families, cluster_to_candidate, cluster_assignment)
        assert len(rows) == 1
        row = rows[0]
        assert row["label"] == "POSITIVE"
        assert row["supporting_subjects"] == 2
        # Two supporting sequences: explicit-echo rule needs >= 3 sequences.
        assert row["wording_status"] == WordingStatus.INFERRED_PARAPHRASE.value
        # Support 2 < 3: MEDIUM needs support >= 3, so the result is LOW.
        assert row["confidence"] == "LOW"
        assert row["order"] == 1

    def test_mixed_assigned_when_share_low(self):
        records = [
            _make_inference_record(
                "S0001",
                [
                    _episode("S0001", 1, "POSITIVE", "HIGH"),
                    _episode("S0001", 2, "NEGATIVE", "HIGH"),
                ],
                "p",
                "s",
                "c",
            )
        ]
        candidates = _candidate_rows(records)
        cluster_to_candidate = {"b1-c1": [c["candidate_id"] for c in candidates]}
        families = [
            {
                "family_id": "f1",
                "question_tr": "q",
                "question_en": "q",
                "member_cluster_ids": ["b1-c1"],
            }
        ]
        rows = aggregate_families(
            candidates, records, families, cluster_to_candidate, {"b1-c1": "f1"}
        )
        assert rows[0]["label"] == "MIXED"

    def test_neutral_tie_break(self):
        records = [
            _make_inference_record(
                "S0001",
                [
                    _episode("S0001", 1, "NEGATIVE", "HIGH"),
                    _episode("S0001", 2, "POSITIVE", "HIGH"),
                    _episode("S0001", 3, "NEUTRAL", "HIGH"),
                ],
                "p",
                "s",
                "c",
            )
        ]
        candidates = _candidate_rows(records)
        cluster_to_candidate = {"b1-c1": [c["candidate_id"] for c in candidates]}
        families = [
            {
                "family_id": "f1",
                "question_tr": "q",
                "question_en": "q",
                "member_cluster_ids": ["b1-c1"],
            }
        ]
        rows = aggregate_families(
            candidates, records, families, cluster_to_candidate, {"b1-c1": "f1"}
        )
        assert rows[0]["label"] == "MIXED"

    def test_confidence_levels(self):
        # LOW confidence: support < 3 or agreement < 0.60.
        records = [
            _make_inference_record(
                "S0001",
                [
                    _episode("S0001", 1, "POSITIVE", "LOW"),
                    _episode("S0001", 2, "NEGATIVE", "LOW"),
                ],
                "p",
                "s",
                "c",
            )
        ]
        candidates = _candidate_rows(records)
        cluster_to_candidate = {"b1-c1": [c["candidate_id"] for c in candidates]}
        families = [
            {
                "family_id": "f1",
                "question_tr": "q",
                "question_en": "q",
                "member_cluster_ids": ["b1-c1"],
            }
        ]
        rows = aggregate_families(
            candidates, records, families, cluster_to_candidate, {"b1-c1": "f1"}
        )
        assert rows[0]["confidence"] == "LOW"

    def test_order_ranks_by_median_position(self):
        records = [
            _make_inference_record(
                "S0001",
                [_episode("S0001", 1, "NEUTRAL", "MEDIUM"), _episode("S0001", 2, "NEUTRAL", "MEDIUM")],
                "p",
                "s",
                "c",
            ),
            _make_inference_record(
                "S0002",
                [_episode("S0002", 1, "NEUTRAL", "MEDIUM")],
                "p",
                "s",
                "c",
            ),
        ]
        candidates = _candidate_rows(records)
        by_seq = {}
        for c in candidates:
            by_seq.setdefault(c["sequence_id"], []).append(c)
        cluster_to_candidate = {}
        families = []
        index = 0
        for seq_id in ("S0001", "S0002"):
            index += 1
            cluster_id = f"b1-c{index}"
            cluster_to_candidate[cluster_id] = [c["candidate_id"] for c in by_seq[seq_id]]
            families.append(
                {
                    "family_id": f"f{index}",
                    "question_tr": f"q {index}",
                    "question_en": f"q {index}",
                    "member_cluster_ids": [cluster_id],
                }
            )
        cluster_assignment = {cid: f"f{i+1}" for i, cid in enumerate(cluster_to_candidate)}
        rows = aggregate_families(
            candidates, records, families, cluster_to_candidate, cluster_assignment
        )
        # S0001 median normalized position = 0.75; S0002 = 1.0 -> S0001 first.
        assert rows[0]["question_tr"] == "q 1"
        assert rows[1]["question_tr"] == "q 2"


class TestRenderAndPrivacy:
    def _render(self, tmp_path):
        records, candidates, families, cluster_to_candidate, cluster_assignment = _family_setup()
        rows = aggregate_families(candidates, records, families, cluster_to_candidate, cluster_assignment)
        run_dir = tmp_path / "run"
        return render_tables(
            rows,
            run_dir=run_dir,
            deployment_id="qwen38_test",
            source_commit="c",
        ), run_dir

    def test_render_deterministic(self, tmp_path):
        result, run_dir = self._render(tmp_path)
        assert result["rows"] == 1
        expected_hash = _sha256_text(
            json.dumps(
                load_table_rows(run_dir / "turkish_inferred_questions.json"),
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        assert result["rows_sha256"] == expected_hash

    def test_render_tables_agree(self, tmp_path):
        result, run_dir = self._render(tmp_path)
        csv_rows = load_table_rows(run_dir / "turkish_inferred_questions.csv", "csv")
        json_rows = load_table_rows(run_dir / "turkish_inferred_questions.json", "json")
        md_rows = load_table_rows(run_dir / "turkish_inferred_questions.md", "md")
        assert csv_rows == json_rows == md_rows
        assert list(csv_rows[0].keys()) == list(FINAL_TABLE_COLUMNS)

    def test_no_window_marker_in_outputs(self, tmp_path):
        result, run_dir = self._render(tmp_path)
        for fmt in ("csv", "json", "md"):
            for row in load_table_rows(run_dir / f"turkish_inferred_questions.{fmt}", fmt):
                for column in FINAL_TABLE_COLUMNS:
                    assert "[WINDOW" not in str(row[column])

    def test_deterministic_rendering_hashing(self, tmp_path):
        records, candidates, families, cluster_to_candidate, cluster_assignment = _family_setup()
        rows = aggregate_families(candidates, records, families, cluster_to_candidate, cluster_assignment)
        run_dir_a = tmp_path / "a"
        run_dir_b = tmp_path / "b"
        result_a = render_tables(rows, run_dir=run_dir_a, deployment_id="d", source_commit="c")
        result_b = render_tables(rows, run_dir=run_dir_b, deployment_id="d", source_commit="c")
        assert result_a["rows_sha256"] == result_b["rows_sha256"]
        assert (run_dir_a / "turkish_inferred_questions.csv").read_text(encoding="utf-8") == (
            run_dir_b / "turkish_inferred_questions.csv"
        ).read_text(encoding="utf-8")


class TestAuditPrivacy:
    def test_overlap_detection(self):
        window = "Bugün sabah erkenden pazara gittim ve taze meyve aldım sonra da eve döndüm."
        text = "Bugün sabah erkenden pazara gittim ve taze meyve aldım sonra da eve döndüm, aynen böyle oldu."
        assert len(window.split()) >= 12
        assert ngram_overlap_at_least(text, [window], 12)
        assert not ngram_overlap_at_least("Tamamen farklı bir konu hakkında konuşuyorum bugün", [window], 12)

    def test_compact_texts_no_leakage(self, tmp_path):
        records, candidates, families, cluster_to_candidate, cluster_assignment = _family_setup()
        rows = aggregate_families(candidates, records, families, cluster_to_candidate, cluster_assignment)
        run_dir = tmp_path / "run"
        render_tables(rows, run_dir=run_dir, deployment_id="d", source_commit="c")
        compact = _compact_texts(
            run_dir / "turkish_inferred_questions.csv",
            run_dir / "turkish_inferred_questions.json",
            run_dir / "turkish_inferred_questions.md",
        )
        assert compact["csv_rows"] == compact["json_rows"] == compact["md_rows"]
        for text in compact["texts"]:
            assert "S0001" not in text
            assert "[WINDOW" not in text


class TestRemoteAuditReference:
    def test_local_audit_requires_matching_remote_audit_sidecar(self, tmp_path, monkeypatch):
        import src.qwen38.audit as audit_module

        transcript = tmp_path / "transcript.jsonl"
        transcript_payload = json.dumps({
            "audio_path": "/private/subA-1-1-ank.wav",
            "transcript": "short source text",
        }, ensure_ascii=False) + "\n"
        transcript.write_text(transcript_payload, encoding="utf-8")
        source_hash = _sha256_text(transcript_payload)
        monkeypatch.setattr(audit_module, "TURKISH_SOURCE_HASH", source_hash)

        deploy = tmp_path / "deploy"
        deployment_id = "deployment"
        env_dir = deploy / deployment_id / "environment"
        env_dir.mkdir(parents=True)
        runtime = dict(ENV_PINS)
        runtime.update({"model_id": MODEL_ID, "model_revision": MODEL_REVISION})
        (env_dir / "runtime_versions.json").write_text(json.dumps(runtime) + "\n", encoding="utf-8")

        model_dir = tmp_path / "model"
        model_dir.mkdir()
        (model_dir / "config.json").write_bytes(b"model")
        model_hash = _sha256_text("model")
        (model_dir / "SHA256SUMS").write_text(f"{model_hash}  config.json\n", encoding="utf-8")
        wheelhouse_dir = tmp_path / "wheelhouse"
        wheelhouse_dir.mkdir()
        (wheelhouse_dir / "package.whl").write_bytes(b"wheel")
        wheel_hash = _sha256_text("wheel")
        (wheelhouse_dir / "SHA256SUMS").write_text(f"{wheel_hash}  package.whl\n", encoding="utf-8")

        source_commit = "d" * 40
        selection_path = deploy / deployment_id / "serving_selection_v2.json"
        selection_path.parent.mkdir(parents=True, exist_ok=True)
        selection_path.write_text(json.dumps({
            "selection_version": 2,
            "selected_tp": 2,
            "source_commit": source_commit,
            "selection_implementation_commit": source_commit,
        }) + "\n", encoding="utf-8")
        selection_hash = _sha256_file_for_test(selection_path)

        run_id = "q38tr_dddddddddddd_attempt1"
        run_dir = tmp_path / "run"
        records, candidates, families, cluster_to_candidate, cluster_assignment = _family_setup()
        rows = aggregate_families(candidates, records, families, cluster_to_candidate, cluster_assignment)
        render_tables(rows, run_dir=run_dir, deployment_id=deployment_id, source_commit=source_commit)
        manifest = {
            "turkish_run_id": run_id,
            "analysis_attempt": 1,
            "source_sha256": source_hash,
            "source_commit": source_commit,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "prompt_version": PROMPT_VERSION,
            "prompt_contract_sha256": prompt_contract_sha256(),
            "generation_settings_hash": generation_settings_hash(TURKISH_MAX_TOKENS),
            "request_settings": request_settings(TURKISH_MAX_TOKENS),
            "selected_tp": 2,
            "selection_file_sha256": selection_hash,
        }
        manifest_path = run_dir / "run_manifest.json"
        manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")
        manifest_hash = _sha256_file_for_test(manifest_path)

        compact_hashes = {
            name: _sha256_file_for_test(run_dir / name)
            for name in (
                "turkish_inferred_questions.csv",
                "turkish_inferred_questions.json",
                "turkish_inferred_questions.md",
            )
        }
        remote_audit = tmp_path / "remote_audit.json"
        remote_audit.write_text(json.dumps({
            "passed": True,
            "compact_artifact_hashes": compact_hashes,
            "run_manifest_sha256": manifest_hash,
            "selection_file_sha256": selection_hash,
        }) + "\n", encoding="utf-8")
        remote_sidecar = tmp_path / "remote_audit.json.sha256"
        remote_sidecar.write_text(
            f"{_sha256_file_for_test(remote_audit)}  remote_audit.json\n", encoding="utf-8"
        )
        slurm = {
            "job_id": "123",
            "state": "COMPLETED",
            "exit_code": "0:0",
            "node": "node",
            "start_time": "2026-08-19T00:00:00",
            "end_time": "2026-08-19T00:01:00",
            "turkish_run_id": run_id,
            "source_commit": source_commit,
        }
        result = audit_turkish(
            run_dir,
            turkish_run_id=run_id,
            transcript_path=transcript,
            deploy_dir=deploy,
            deployment_id=deployment_id,
            model_dir=model_dir,
            wheelhouse_dir=wheelhouse_dir,
            source_commit=source_commit,
            selection_file=selection_path,
            slurm_metadata=slurm,
            remote_reference=remote_audit,
            remote_audit_sha256_path=remote_sidecar,
        )
        checks = {check["check_id"]: check for check in result["checks"]}
        assert checks["remote_audit_reference"]["passed"] is True
        assert checks["remote_compact_hashes"]["passed"] is True
        assert checks["remote_manifest_hash"]["passed"] is True
        assert checks["remote_selection_hash"]["passed"] is True

        remote_sidecar.write_text("0" * 64 + "  remote_audit.json\n", encoding="utf-8")
        failed = audit_turkish(
            run_dir,
            turkish_run_id=run_id,
            transcript_path=transcript,
            deploy_dir=deploy,
            deployment_id=deployment_id,
            model_dir=model_dir,
            wheelhouse_dir=wheelhouse_dir,
            source_commit=source_commit,
            selection_file=selection_path,
            slurm_metadata=slurm,
            remote_reference=remote_audit,
            remote_audit_sha256_path=remote_sidecar,
        )
        failed_checks = {check["check_id"]: check for check in failed["checks"]}
        assert failed_checks["remote_audit_reference"]["passed"] is False


def _small_restricted_run(tmp_path):
    """Build a two-sequence restricted bundle for aggregation-audit tests."""
    from src.qwen38.turkish_questions import _sha256_file

    run_dir = tmp_path / "restricted_run"
    restricted = run_dir / "restricted"
    inferences = restricted / "subject_inferences"
    consolidation = restricted / "consolidation_batches"
    inferences.mkdir(parents=True)
    consolidation.mkdir(parents=True)
    commit = "c" * 40
    run_id = "q38tr_cccccccccccc_attempt1"
    manifest = {
        "turkish_run_id": run_id,
        "analysis_attempt": 1,
        "deployment_id": "deployment",
        "source_sha256": TURKISH_SOURCE_HASH,
        "source_commit": commit,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "prompt_version": PROMPT_VERSION,
        "prompt_contract_sha256": prompt_contract_sha256(),
        "generation_settings_hash": generation_settings_hash(TURKISH_MAX_TOKENS),
        "request_settings": request_settings(TURKISH_MAX_TOKENS),
        "expected_windows": 2,
        "expected_sequences": 2,
        "consolidation_batches": [1, 1],
    }
    manifest_path = run_dir / "run_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    manifest_hash = _sha256_file(manifest_path)

    sequences = []
    records = []
    for index in (1, 2):
        sequence_id = f"S{index:04d}"
        user_prompt = f"Sequence id: {sequence_id}\n\n[WINDOW 1]\nanswer {index}"
        components = prompt_component_hashes(user_prompt)
        sequence = {
            "turkish_run_id": run_id,
            "analysis_attempt": 1,
            "deployment_id": "deployment",
            "sequence_id": sequence_id,
            "window_count": 1,
            "windows": [{"window": 1, "text": f"answer {index}"}],
            "user_prompt": user_prompt,
            "source_sha256": TURKISH_SOURCE_HASH,
            "source_commit": commit,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "prompt_version": PROMPT_VERSION,
            "generation_settings_hash": generation_settings_hash(TURKISH_MAX_TOKENS),
            "run_manifest_sha256": manifest_hash,
            **components,
        }
        sequences.append(sequence)
        episode = {
            "sequence_id": sequence_id,
            "episode_order": 1,
            "question_tr": "Hangi konu hakkinda konusuldu?",
            "question_en": "What topic was discussed?",
            "label": "NEUTRAL",
            "wording_status": "INFERRED_PARAPHRASE",
            "confidence": "LOW",
            "evidence_window_indices": [1],
            "evidence_basis": "topic framing",
            "abstain_reason": "",
        }
        record = dict(_episode_provenance(sequence))
        record.update({
            "sequence_id": sequence_id,
            "status": "completed",
            "episode_count": 1,
            "episodes": [episode],
        })
        records.append(record)

    (restricted / "prepared_sequences.jsonl").write_text(
        "".join(json.dumps(sequence, ensure_ascii=False) + "\n" for sequence in sequences),
        encoding="utf-8",
    )
    for record in records:
        (inferences / f"{record['sequence_id']}.json").write_text(
            json.dumps(record, ensure_ascii=False), encoding="utf-8"
        )

    clusters = []
    cluster_to_candidate = {}
    for index in (1, 2):
        sequence_id = f"S{index:04d}"
        cluster_id = f"c{index}"
        candidate_id = f"{sequence_id}-e1"
        clusters.append(cluster_id)
        cluster_to_candidate[cluster_id] = [candidate_id]
        batch = {
            "batch_index": index,
            "sequence_ids": [sequence_id],
            "candidate_count": 1,
            "clusters": [{
                "cluster_id": cluster_id,
                "canonical_question_tr": "Hangi konu hakkinda konusuldu?",
                "canonical_question_en": "What topic was discussed?",
                "member_candidate_ids": [candidate_id],
            }],
            "assignment": {candidate_id: cluster_id},
        }
        (consolidation / f"batch_{index:02d}.json").write_text(
            json.dumps(batch, ensure_ascii=False), encoding="utf-8"
        )
    final_merge = {
        "families": [{
            "family_id": "f1",
            "question_tr": "Hangi konu hakkinda konusuldu?",
            "question_en": "What topic was discussed?",
            "member_cluster_ids": clusters,
        }],
        "cluster_assignment": {cluster_id: "f1" for cluster_id in clusters},
    }
    (consolidation / "final_merge.json").write_text(
        json.dumps(final_merge, ensure_ascii=False), encoding="utf-8"
    )
    candidates = collect_candidates(records)
    rows = aggregate_families(
        candidates,
        records,
        final_merge["families"],
        cluster_to_candidate,
        final_merge["cluster_assignment"],
    )
    render_tables(rows, run_dir=run_dir, deployment_id="deployment", source_commit=commit)
    return run_dir, manifest


class TestRestrictedAggregationAudit:
    def test_recomputation_detects_altered_final_cell(self, tmp_path):
        run_dir, manifest = _small_restricted_run(tmp_path)
        compact = _compact_texts(
            run_dir / "turkish_inferred_questions.csv",
            run_dir / "turkish_inferred_questions.json",
            run_dir / "turkish_inferred_questions.md",
        )
        result = _recompute_restricted_evidence(run_dir, manifest, compact=compact)
        assert result["completed_subject_files"] == 2
        payload = json.loads((run_dir / "turkish_inferred_questions.json").read_text(encoding="utf-8"))
        payload["rows"][0]["question_en"] = "altered"
        (run_dir / "turkish_inferred_questions.json").write_text(
            json.dumps(payload, ensure_ascii=False), encoding="utf-8"
        )
        altered = _compact_texts(
            run_dir / "turkish_inferred_questions.csv",
            run_dir / "turkish_inferred_questions.json",
            run_dir / "turkish_inferred_questions.md",
        )
        with pytest.raises(ValueError, match="recomputed rows differ"):
            _recompute_restricted_evidence(run_dir, manifest, compact=altered)


class TestStemParsingRoundTrip:
    def test_stems_match_prepare(self, tmp_path):
        transcript = tmp_path / "t.jsonl"
        source_hash = _write_transcript(
            transcript, {"kisi-1": [1, 2], "kisi-2": [1]}, transcripts={
                "kisi-1": {1: "metin bir", 2: "metin iki"},
                "kisi-2": {1: "metin uc"},
            }
        )
        run_dir = tmp_path / "run"
        prepare_sequences(
            transcript,
            run_dir=run_dir,
            deployment_id="d",
            model_revision="r",
            source_commit="c",
            source_sha256=source_hash,
            expected_sequences=2,
            expected_windows=3,
        )
        for sequence in load_prepared_sequences(run_dir / "restricted" / "prepared_sequences.jsonl"):
            assert sequence["window_count"] == len(sequence["windows"])
            for window in sequence["windows"]:
                assert parse_filename_stem(f"x-1-{window['window']}-ank") is not None
