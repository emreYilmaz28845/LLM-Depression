"""Turkish question-recovery pipeline tests: prepare, resume, consolidation,
aggregation rules, deterministic rendering, and privacy checks."""
from __future__ import annotations

import hashlib
import json

import pytest

from src.qwen38.audit import (
    _compact_texts,
    audit_turkish,
    ngram_overlap_at_least,
)
from src.qwen38.contracts import (
    FINAL_TABLE_COLUMNS,
    WordingStatus,
    generation_settings_hash,
    parse_filename_stem,
)
from src.qwen38.turkish_questions import (
    _check_cluster_assignment,
    _check_family_assignment,
    aggregate_families,
    collect_candidates,
    load_prepared_sequences,
    load_table_rows,
    prepare_sequences,
    render_tables,
)

FIXTURE = "tests/fixtures/qwen38_synthetic_cases.jsonl"


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


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


def _make_inference_record(sequence_id, episodes, prompt_hash, source_sha256, source_commit, model_revision="1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"):
    return {
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
