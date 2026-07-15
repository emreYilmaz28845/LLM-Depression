from __future__ import annotations

import copy
import unittest

from src import e0_perturbations as e0


def _config() -> dict:
    return {
        "dataset": "daic",
        "data": {
            "use_audio": True,
            "use_text": True,
            "sample_mode": "subject_audio",
            "chunks_per_subject": 4,
        },
    }


def _subject(subject_id: str, num_chunks: int) -> tuple[dict, list[dict]]:
    rows = [
        {
            "subject_id": subject_id,
            "sample_id": f"{subject_id}_random_segment_{ordinal}",
            "chunk_id": f"random_segment_{ordinal}",
            "audio_path": f"/{subject_id}/random_segment_{ordinal}.wav",
        }
        for ordinal in range(1, num_chunks + 1)
    ]
    # Reproduce the legacy runtime's lexical source order and baked selection.
    lexical_rows = sorted(rows, key=lambda row: row["sample_id"])
    step = (num_chunks - 1) / 3
    legacy_indices = [int(round(index * step)) for index in range(4)]
    legacy_paths = [lexical_rows[index]["audio_path"] for index in legacy_indices]
    example = {
        "dataset": "daic",
        "subject_id": subject_id,
        "sample_id": subject_id,
        "label": 1,
        "label_text": "Depressed",
        "internal_label_text": "Depressed",
        "transcript": f"fixed transcript {subject_id}",
        "audio_paths": legacy_paths,
        "audio_clip_seconds": [30.0] * 4,
        "subject_chunk_paths": [row["audio_path"] for row in lexical_rows],
        "chunks_per_subject": 4,
        "input_modality": "audio_text",
        "prompt_text": f"fixed prompt {subject_id}",
        "training_text": f"fixed prompt {subject_id}Depressed",
        "question_id": "subject_audio_bundle",
    }
    return example, lexical_rows


class NumericOrderingTests(unittest.TestCase):
    def test_numeric_suffix_order_is_not_lexical_order(self) -> None:
        _, rows = _subject("300", 10)
        ordered = e0._numeric_order_subject_rows(rows)
        self.assertEqual(
            [row["chunk_id"] for row in ordered],
            [f"random_segment_{index}" for index in range(1, 11)],
        )

    def test_ambiguous_duplicate_numeric_ordinals_fail_closed(self) -> None:
        rows = [
            {
                "subject_id": "300",
                "sample_id": "300_a_1",
                "chunk_id": "a_1",
                "audio_path": "/300/a.wav",
            },
            {
                "subject_id": "300",
                "sample_id": "300_b_1",
                "chunk_id": "b_1",
                "audio_path": "/300/b.wav",
            },
        ]
        with self.assertRaisesRegex(ValueError, "Duplicate numeric chunk ordinals"):
            e0._numeric_order_subject_rows(rows)


class NumericScheduleTests(unittest.TestCase):
    def test_actual_daic_pool_sizes_have_pinned_balanced_schedules(self) -> None:
        expected = {
            10: (
                1,
                "0afddc50dec04eb878ec3e89ee3bbf24d8299618b0bfa573d1f1563afccd633c",
                [4, 4, 3, 3, 3, 3, 3, 3, 3, 3],
            ),
            15: (
                7,
                "95dafc1a158786741e7a4d6d5a04f2e95155ef72b9fee4ced18ab9035ee0e6ce",
                [3, 2, 2, 2, 2, 2, 2, 3, 2, 2, 2, 2, 2, 2, 2],
            ),
        }
        for num_chunks, (step, digest, counts) in expected.items():
            schedule = e0._numeric_balanced_schedule(num_chunks)
            self.assertEqual(schedule["modular_step"], step)
            self.assertEqual(schedule["schedule_sha256"], digest)
            self.assertEqual(schedule["exposure_counts_by_ordinal"], counts)
            selections = schedule["view_ordinal_indices_zero_based"]
            self.assertEqual(len(selections), 8)
            self.assertEqual(len({frozenset(row) for row in selections}), 8)
            self.assertTrue(all(row == sorted(row) and len(row) == 4 for row in selections))
            self.assertLessEqual(max(counts) - min(counts), 1)
            self.assertEqual(sum(counts), 32)

    def test_too_few_distinct_k4_bundles_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "Could not construct eight distinct"):
            e0._numeric_balanced_schedule(4)


class NumericMaterializationTests(unittest.TestCase):
    def setUp(self) -> None:
        first, first_rows = _subject("300", 10)
        second, second_rows = _subject("350", 15)
        self.examples = [first, second]
        self.rows = first_rows + second_rows

    def test_all_eight_views_are_real_distinct_balanced_numeric_bundles(self) -> None:
        originals = copy.deepcopy(self.examples)
        selections_by_subject = {"300": [], "350": []}
        reference_plan_sha = None
        for view_index in range(8):
            materialized, view = e0.materialize_numeric_balanced_view(
                self.examples,
                self.rows,
                _config(),
                expected_k=4,
                view_index=view_index,
            )
            self.assertEqual(view["view_family"], e0.NUMERIC_BALANCED_VIEW_FAMILY)
            self.assertEqual(view["view_index"], view_index)
            self.assertEqual(view["available_views"], 8)
            self.assertFalse(view["labels_or_content_used_for_selection"])
            self.assertIn("not provide timestamps", view["chronology_caveat"])
            if reference_plan_sha is None:
                reference_plan_sha = view["selection_schedule_sha256"]
            self.assertEqual(view["selection_schedule_sha256"], reference_plan_sha)

            for source, changed in zip(self.examples, materialized):
                subject_id = source["subject_id"]
                selected_ids = view["selected_sample_ids_by_subject"][subject_id]
                selections_by_subject[subject_id].append(frozenset(selected_ids))
                numeric_suffixes = [
                    int(path.rsplit("_", 1)[1].removesuffix(".wav"))
                    for path in changed["audio_paths"]
                ]
                self.assertEqual(numeric_suffixes, sorted(numeric_suffixes))
                self.assertEqual(len(numeric_suffixes), 4)
                self.assertEqual(changed["prompt_text"], source["prompt_text"])
                self.assertEqual(changed["transcript"], source["transcript"])
                self.assertEqual(changed["training_text"], source["training_text"])

        for subject_id, selections in selections_by_subject.items():
            self.assertEqual(len(set(selections)), 8, subject_id)
        self.assertEqual(self.examples, originals)
        # Numeric view 0 is deliberately not the old lexical replication bundle.
        numeric_zero, _ = e0.materialize_numeric_balanced_view(
            self.examples,
            self.rows,
            _config(),
            expected_k=4,
            view_index=0,
        )
        self.assertNotEqual(numeric_zero[0]["audio_paths"], self.examples[0]["audio_paths"])

    def test_materialization_is_exactly_deterministic_and_range_checked(self) -> None:
        first = e0.materialize_numeric_balanced_view(
            self.examples,
            self.rows,
            _config(),
            expected_k=4,
            view_index=3,
        )
        second = e0.materialize_numeric_balanced_view(
            self.examples,
            list(reversed(self.rows)),
            _config(),
            expected_k=4,
            view_index=3,
        )
        self.assertEqual(first, second)
        with self.assertRaisesRegex(ValueError, "view_index must be"):
            e0.materialize_numeric_balanced_view(
                self.examples,
                self.rows,
                _config(),
                expected_k=4,
                view_index=8,
            )

    def test_cli_defaults_to_legacy_family(self) -> None:
        args = e0.parse_args(["--checkpoint-dir", "/tmp/checkpoint"])
        self.assertEqual(args.view_family, e0.LEGACY_VIEW_FAMILY)
        self.assertEqual(args.view_index, 0)


if __name__ == "__main__":
    unittest.main()
