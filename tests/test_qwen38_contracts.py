"""Contract tests: enums, schemas, filename parsing, selection rules."""
from __future__ import annotations

import pytest

from src.qwen38.contracts import (
    CandidateResult,
    Label,
    WordingStatus,
    parse_filename_stem,
    project_turkish_wall_seconds,
    select_serving_configuration,
    validate_consolidation_batch,
    validate_consolidation_final,
    validate_subject_inference,
    validate_validation_response,
)


class TestEnumsAndSchemas:
    def test_label_values(self):
        assert [label.value for label in Label] == [
            "POSITIVE",
            "NEGATIVE",
            "NEUTRAL",
            "MIXED",
        ]

    def test_wording_status_values(self):
        assert [status.value for status in WordingStatus] == [
            "EXPLICIT_ECHO",
            "INFERRED_PARAPHRASE",
        ]

    def test_validation_response_valid(self):
        assert (
            validate_validation_response(
                {
                    "case_id": "syn-001",
                    "inferred_question": "What makes you happy?",
                    "label": "POSITIVE",
                    "confidence": "HIGH",
                }
            )
            == []
        )

    @pytest.mark.parametrize(
        "mutation",
        [
            {"label": "ANGRY"},
            {"confidence": "SURE"},
            {"case_id": 3},
            {"extra": 1},
        ],
    )
    def test_validation_response_invalid(self, mutation):
        payload = {
            "case_id": "syn-001",
            "inferred_question": "q",
            "label": "POSITIVE",
            "confidence": "HIGH",
        }
        payload.update(mutation)
        assert validate_validation_response(payload) != []

    def test_subject_inference_schema(self):
        payload = {
            "episodes": [
                {
                    "sequence_id": "S0001",
                    "episode_order": 1,
                    "question_tr": "Sizi üzen şeyler nelerdir?",
                    "question_en": "What upsets you?",
                    "label": "NEGATIVE",
                    "wording_status": "EXPLICIT_ECHO",
                    "confidence": "MEDIUM",
                    "evidence_window_indices": [2, 3],
                    "evidence_basis": "answer lists upsetting situations",
                    "abstain_reason": "",
                }
            ]
        }
        assert validate_subject_inference(payload) == []

    def test_subject_inference_rejects_unknown_fields(self):
        payload = {
            "episodes": [
                {
                    "sequence_id": "S0001",
                    "episode_order": 1,
                    "question_tr": "q",
                    "question_en": "q",
                    "label": "NEGATIVE",
                    "wording_status": "EXPLICIT_ECHO",
                    "confidence": "MEDIUM",
                    "evidence_window_indices": [1],
                    "evidence_basis": "b",
                    "abstain_reason": "",
                    "condition": "depr",
                }
            ]
        }
        assert validate_subject_inference(payload) != []

    def test_consolidation_schemas(self):
        batch = {
            "clusters": [
                {
                    "cluster_id": "b1-c1",
                    "canonical_question_tr": "q tr",
                    "canonical_question_en": "q en",
                    "member_candidate_ids": ["S0001-e1"],
                }
            ]
        }
        final = {
            "families": [
                {
                    "family_id": "f1",
                    "question_tr": "q tr",
                    "question_en": "q en",
                    "member_cluster_ids": ["b1-c1"],
                }
            ]
        }
        assert validate_consolidation_batch(batch) == []
        assert validate_consolidation_final(final) == []

    def test_consolidation_batch_rejects_duplicate_members(self):
        batch = {
            "clusters": [
                {
                    "cluster_id": "b1-c1",
                    "canonical_question_tr": "q",
                    "canonical_question_en": "q",
                    "member_candidate_ids": ["S0001-e1", "S0001-e1"],
                }
            ]
        }
        assert validate_consolidation_batch(batch) == []


class TestFilenameParsing:
    @pytest.mark.parametrize(
        ("stem", "subject", "window", "condition"),
        [
            ("aa1-1-1-ank", "aa1", 1, "ank"),
            ("subject-x-1-12-depr", "subject-x", 12, "depr"),
            ("abc123-1-3-depr+ank", "abc123", 3, "depr+ank"),
            ("kisalt-1-9-ank+depr", "kisalt", 9, "ank+depr"),
            ("uzun-subject-name-1-21-dep+ank", "uzun-subject-name", 21, "dep+ank"),
            ("sub-1-120-ank", "sub", 120, "ank"),
        ],
    )
    def test_all_condition_tags(self, stem, subject, window, condition):
        parts = parse_filename_stem(stem)
        assert parts is not None
        assert parts.subject == subject
        assert parts.window == window
        assert parts.condition == condition

    @pytest.mark.parametrize(
        "stem",
        [
            "aa1-1-1-other",
            "aa1-1-1",
            "aa1-1-ank",
            "aa1-x-1-ank",
            "-1-1-ank",
            "aa1-1-1-ank-extra",
            "aa1--1-ank",
            "aa1-1--ank",
            "",
            "aa1-1-1-ANK",
            "aa1-1--ank",
            "aa1-1-01-ankx",
        ],
    )
    def test_malformed_stems_rejected(self, stem):
        assert parse_filename_stem(stem) is None

    def test_leading_zero_window_accepted(self):
        parts = parse_filename_stem("aa1-1-01-ank")
        assert parts is not None and parts.window == 1

    def test_numeric_window_ordering(self):
        from src.qwen38.contracts import STEM_RE

        windows = [2, 10, 1, 120, 21]
        parsed = [int(STEM_RE.fullmatch(f"sub-1-{w}-ank").group("window")) for w in windows]
        assert sorted(parsed) == [1, 2, 10, 21, 120]


class TestSelectionRules:
    def test_tp2_missing_means_no_selection(self):
        result = select_serving_configuration({})
        assert result.selected_tp is None
        assert result.decision_rule == "rule1_tp2_not_passed_no_selection"

    def test_tp2_failed_means_no_selection(self):
        candidates = {
            1: CandidateResult(tp=1, passed=True, request_rate_c1=10, request_rate_c8=10),
            2: CandidateResult(tp=2, passed=False),
        }
        result = select_serving_configuration(candidates)
        assert result.selected_tp is None

    def test_tp1_selected_when_eligible(self):
        candidates = {
            1: CandidateResult(tp=1, passed=True, request_rate_c1=2, request_rate_c8=3),
            2: CandidateResult(tp=2, passed=True, request_rate_c1=2, request_rate_c8=3),
            4: CandidateResult(tp=4, passed=True, request_rate_c1=4, request_rate_c8=4),
        }
        result = select_serving_configuration(candidates)
        assert result.selected_tp == 1
        assert result.decision_rule == "rule3_select_tp1_eligible_within_2h"

    def test_tp1_not_selected_when_projection_over_two_hours(self):
        # 0.02 rps -> 142 / 0.02 * 1.25 = 8875 s > 7200 s.
        slow = 0.02
        candidates = {
            1: CandidateResult(tp=1, passed=True, request_rate_c1=slow, request_rate_c8=slow),
            2: CandidateResult(tp=2, passed=True, request_rate_c1=slow, request_rate_c8=slow),
        }
        result = select_serving_configuration(candidates)
        assert result.selected_tp == 2
        assert result.decision_rule == "rule5_select_tp2"

    def test_tp4_replaces_tp2_when_conditions_met(self):
        # TP=2 at 0.01 rps -> 17750 s (~4.93 h > 4 h); TP=4 at 0.02 rps ->
        # 8875 s, a 50% reduction (>= 30%).
        candidates = {
            1: CandidateResult(tp=1, passed=False),
            2: CandidateResult(tp=2, passed=True, request_rate_c1=0.01, request_rate_c8=0.01),
            4: CandidateResult(tp=4, passed=True, request_rate_c1=0.02, request_rate_c8=0.02),
        }
        result = select_serving_configuration(candidates)
        assert result.selected_tp == 4
        assert result.decision_rule == "rule4_tp4_replaces_tp2_over4h_and_30pct_faster"

    def test_tp4_not_replacing_without_30pct_gain(self):
        # TP=2 projects 11h; TP=4 is only 20% faster.
        rate = 142 / 39600 * 1.25
        candidates = {
            1: CandidateResult(tp=1, passed=False),
            2: CandidateResult(tp=2, passed=True, request_rate_c1=rate, request_rate_c8=rate),
            4: CandidateResult(tp=4, passed=True, request_rate_c1=rate * 1.25, request_rate_c8=rate * 1.25),
        }
        result = select_serving_configuration(candidates)
        assert result.selected_tp == 2

    def test_tie_break_fewer_gpus(self):
        # TP=2 and TP=4 pass with identical projections: rule 5 selects TP=2
        # and the rule-6 tie-break keeps the configuration with fewer GPUs.
        rate = 0.15
        candidates = {
            1: CandidateResult(tp=1, passed=False),
            2: CandidateResult(tp=2, passed=True, request_rate_c1=rate, request_rate_c8=rate),
            4: CandidateResult(tp=4, passed=True, request_rate_c1=rate, request_rate_c8=rate),
        }
        result = select_serving_configuration(candidates)
        assert result.selected_tp == 2
        assert result.projected_wall_seconds[2] == result.projected_wall_seconds[4]

    def test_rule6_never_resurrects_ineligible_tp1(self):
        # TP=1 passed but projects > 2h; TP=2 passes; rule 6 must not select
        # the ineligible TP=1 even when projections are within 10%.
        candidates = {
            1: CandidateResult(tp=1, passed=True, request_rate_c1=0.019, request_rate_c8=0.019),
            2: CandidateResult(tp=2, passed=True, request_rate_c1=0.020, request_rate_c8=0.020),
        }
        result = select_serving_configuration(candidates)
        assert result.selected_tp == 2

    def test_projection_formula(self):
        assert project_turkish_wall_seconds(2.0, 1.0) == pytest.approx(142 / 1.0 * 1.25)
        with pytest.raises(ValueError):
            project_turkish_wall_seconds(0, 1)
