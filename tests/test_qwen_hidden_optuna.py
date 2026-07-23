from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import yaml

from baselines import qwen_hidden_xgb_optuna as optuna_xgb
from baselines import summarize_qwen_hidden_optuna_stability as stability
from scripts import build_qwen_hidden_optuna_followup_matrix as followup
from src.metrics import classification_metrics
from src.utils import read_json, save_json, write_jsonl


class _FakeTrial:
    def __init__(self):
        self.user_attrs = {}

    def suggest_int(self, name, low, high, step=1):
        return low

    def suggest_float(self, name, low, high, log=False):
        return low

    def set_user_attr(self, key, value):
        self.user_attrs[key] = value


class _FakeClassifier:
    fit_calls = 0

    def fit(self, x, y):
        _FakeClassifier.fit_calls += 1
        return self

    def predict_proba(self, x):
        probs = (x[:, 0] > 0).astype(float) * 0.8 + (x[:, 0] <= 0).astype(float) * 0.2
        return np.column_stack([1.0 - probs, probs])


class OptunaObjectiveTests(unittest.TestCase):
    def test_objective_uses_three_inner_fits_without_loading_final_eval(self):
        train_x = np.asarray(
            [
                [-2.0],
                [-1.5],
                [1.0],
                [1.5],
                [-1.0],
                [-0.5],
                [2.0],
                [2.5],
                [-3.0],
                [-2.5],
                [3.0],
                [3.5],
            ],
            dtype=np.float32,
        )
        train_rows = [
            {"sample_id": f"s{i}", "subject_id": f"p{i // 2}", "label": (i // 2) % 2}
            for i in range(len(train_x))
        ]
        assignments = {
            "folds": [
                {"fold": 0, "train_row_indices": [4, 5, 6, 7, 8, 9, 10, 11], "validation_row_indices": [0, 1, 2, 3]},
                {"fold": 1, "train_row_indices": [0, 1, 2, 3, 8, 9, 10, 11], "validation_row_indices": [4, 5, 6, 7]},
                {"fold": 2, "train_row_indices": [0, 1, 2, 3, 4, 5, 6, 7], "validation_row_indices": [8, 9, 10, 11]},
            ]
        }
        metadata = {
            "dataset": "synthetic",
            "input_modality": "audio_text",
            "condition": "audio_text",
            "fold": 0,
            "checkpoint_dir": "synthetic/best_model",
        }
        trial = _FakeTrial()
        _FakeClassifier.fit_calls = 0
        with mock.patch.object(optuna_xgb, "_classifier", return_value=_FakeClassifier()):
            objective = optuna_xgb.make_objective(
                train_x=train_x,
                train_rows=train_rows,
                metadata=metadata,
                assignments=assignments,
                objective_name="positive_f1",
                fixed_params=optuna_xgb.fixed_xgb_params(seed=1337, xgb_threads=1),
            )
            value = objective(trial)
        self.assertEqual(_FakeClassifier.fit_calls, 3)
        self.assertIn("inner_fold_metrics", trial.user_attrs)
        self.assertIn("inner_oof_metrics", trial.user_attrs)
        self.assertEqual(trial.user_attrs["inner_oof_metrics"]["support_positive"], 3)
        self.assertGreaterEqual(value, 0.0)

    def test_objective_scores_subject_aggregation_not_response_rows(self):
        train_x = np.asarray([[1.0], [-1.0], [-1.0], [-1.0]], dtype=np.float32)
        train_rows = [
            {"sample_id": "a-1", "subject_id": "a", "label": 1},
            {"sample_id": "a-2", "subject_id": "a", "label": 1},
            {"sample_id": "a-3", "subject_id": "a", "label": 1},
            {"sample_id": "b-1", "subject_id": "b", "label": 0},
        ]
        assignments = {
            "folds": [
                {
                    "fold": 0,
                    "train_row_indices": [],
                    "validation_row_indices": [0, 1, 2, 3],
                }
            ]
        }
        metadata = {
            "dataset": "synthetic",
            "input_modality": "audio_text",
            "condition": "audio_text",
            "fold": 0,
        }
        trial = _FakeTrial()
        with mock.patch.object(optuna_xgb, "_classifier", return_value=_FakeClassifier()):
            objective = optuna_xgb.make_objective(
                train_x=train_x,
                train_rows=train_rows,
                metadata=metadata,
                assignments=assignments,
                objective_name="positive_f1",
                fixed_params=optuna_xgb.fixed_xgb_params(seed=1337, xgb_threads=1),
            )
            subject_value = objective(trial)
        response_value = classification_metrics([1, 1, 1, 0], [1, 0, 0, 0])["positive_f1"]
        self.assertEqual(subject_value, 0.0)
        self.assertEqual(response_value, 0.5)

    @unittest.skipIf(importlib.util.find_spec("sklearn") is None, "scikit-learn is not installed")
    def test_inner_subject_assignments_cover_each_subject_once(self):
        rows = []
        for subject_index in range(8):
            for sample_index in range(2):
                rows.append(
                    {
                        "sample_id": f"s{subject_index}-{sample_index}",
                        "subject_id": f"p{subject_index}",
                        "label": subject_index % 2,
                    }
                )
        assignments = optuna_xgb.build_inner_subject_assignments(rows, inner_folds=4, seed=1337)
        validation_subjects = [
            subject_id
            for fold in assignments["folds"]
            for subject_id in fold["validation_subject_ids"]
        ]
        self.assertCountEqual(validation_subjects, [f"p{i}" for i in range(8)])
        for fold in assignments["folds"]:
            self.assertTrue(set(fold["train_subject_ids"]).isdisjoint(fold["validation_subject_ids"]))

    @unittest.skipIf(importlib.util.find_spec("sklearn") is None, "scikit-learn is not installed")
    def test_inner_assignments_are_deterministic_and_keep_all_responses_together(self):
        rows = [
            {
                "sample_id": f"s{subject_index}-{sample_index}",
                "subject_id": f"p{subject_index}",
                "label": subject_index % 2,
            }
            for subject_index in range(12)
            for sample_index in range(3)
        ]
        first = optuna_xgb.build_inner_subject_assignments(rows, inner_folds=3, seed=1337)
        reordered_rows = list(reversed(rows))
        second = optuna_xgb.build_inner_subject_assignments(
            reordered_rows,
            inner_folds=3,
            seed=1337,
        )
        alternate = optuna_xgb.build_inner_subject_assignments(rows, inner_folds=3, seed=7)
        first_validation = {
            subject_id: int(fold["fold"])
            for fold in first["folds"]
            for subject_id in fold["validation_subject_ids"]
        }
        second_validation = {
            subject_id: int(fold["fold"])
            for fold in second["folds"]
            for subject_id in fold["validation_subject_ids"]
        }
        self.assertEqual(first_validation, second_validation)
        alternate_validation = {
            subject_id: int(fold["fold"])
            for fold in alternate["folds"]
            for subject_id in fold["validation_subject_ids"]
        }
        self.assertNotEqual(first_validation, alternate_validation)
        for assignments, source_rows in ((first, rows), (second, reordered_rows)):
            for fold in assignments["folds"]:
                validation_subjects = set(fold["validation_subject_ids"])
                validation_rows = {
                    source_rows[index]["subject_id"] for index in fold["validation_row_indices"]
                }
                self.assertEqual(validation_rows, validation_subjects)


class OptunaConfigTests(unittest.TestCase):
    def test_study_config_sidecar_refuses_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            config = {"schema_version": "test", "objective": "positive_f1"}
            optuna_xgb._write_or_validate_study_config(output, config, "abc")
            self.assertEqual(read_json(output / "study_config.json")["config_sha256"], "abc")
            with self.assertRaisesRegex(ValueError, "differs"):
                optuna_xgb._write_or_validate_study_config(output, config, "def")

    def test_inner_assignment_artifact_refuses_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "inner_subject_assignments.json"
            optuna_xgb._write_or_validate_json({"folds": [1]}, path, path.name)
            optuna_xgb._write_or_validate_json({"folds": [1]}, path, path.name)
            with self.assertRaisesRegex(ValueError, "Refusing to resume"):
                optuna_xgb._write_or_validate_json({"folds": [2]}, path, path.name)

    def test_search_space_matches_declared_bounds(self):
        self.assertEqual(
            optuna_xgb.SEARCH_SPACE,
            {
                "n_estimators": {"kind": "int", "low": 100, "high": 1000, "step": 50},
                "learning_rate": {"kind": "float", "low": 0.005, "high": 0.2, "log": True},
                "max_depth": {"kind": "int", "low": 1, "high": 6},
                "min_child_weight": {"kind": "float", "low": 0.5, "high": 20.0, "log": True},
                "subsample": {"kind": "float", "low": 0.5, "high": 1.0},
                "colsample_bytree": {"kind": "float", "low": 0.1, "high": 1.0},
                "gamma": {"kind": "float", "low": 1e-8, "high": 5.0, "log": True},
                "reg_alpha": {"kind": "float", "low": 1e-8, "high": 20.0, "log": True},
                "reg_lambda": {"kind": "float", "low": 1e-3, "high": 50.0, "log": True},
                "scale_pos_weight": {"kind": "float", "low": 0.25, "high": 4.0, "log": True},
            },
        )
        depth8 = optuna_xgb.resolved_search_space("depth8")
        self.assertEqual(depth8["max_depth"], {"kind": "int", "low": 1, "high": 8})
        self.assertEqual(
            {key: value for key, value in depth8.items() if key != "max_depth"},
            {key: value for key, value in optuna_xgb.SEARCH_SPACE.items() if key != "max_depth"},
        )
        self.assertEqual(optuna_xgb.SEARCH_SPACE["max_depth"]["high"], 6)

    def test_experiment_output_requires_matching_slug_and_fold(self):
        metadata = {"fold": 2}
        valid = Path("/tmp/results/run-name/fold_2/xgb_optuna_raw_t150_d6_seed1337_inner7")
        self.assertEqual(
            optuna_xgb.validate_experiment_output(
                valid,
                metadata=metadata,
                experiment_id=valid.name,
            ),
            "run-name",
        )
        with self.assertRaisesRegex(ValueError, "basename"):
            optuna_xgb.validate_experiment_output(
                valid,
                metadata=metadata,
                experiment_id="different_id",
            )
        with self.assertRaisesRegex(ValueError, "lowercase slug"):
            optuna_xgb.validate_experiment_output(
                valid,
                metadata=metadata,
                experiment_id="Unsafe-ID",
            )
        with self.assertRaisesRegex(ValueError, "fold_2"):
            optuna_xgb.validate_experiment_output(
                Path("/tmp/results/run-name/fold_1/safe_id"),
                metadata=metadata,
                experiment_id="safe_id",
            )

    def test_optuna_raw_matrix_expands_to_expected_no_emotion_jobs(self):
        matrix = yaml.safe_load(Path("configs/features/optuna_raw_matrix.yaml").read_text(encoding="utf-8"))
        rows = [
            (item["dataset"], item.get("condition", item["modality"]), fold, item["objective"])
            for item in matrix["experiments"]
            for fold in item["folds"]
        ]
        self.assertEqual(len(rows), 33)
        self.assertEqual(matrix["expected_jobs"], 33)
        self.assertEqual(sum(1 for dataset, *_ in rows if dataset == "daic"), 3)
        self.assertEqual(sum(1 for dataset, *_ in rows if dataset == "cmdc"), 15)
        self.assertEqual(sum(1 for dataset, *_ in rows if dataset == "turkish"), 15)
        self.assertTrue(all(objective == "positive_f1" for dataset, *_, objective in rows if dataset != "turkish"))
        self.assertTrue(all(objective == "macro_f1" for dataset, *_, objective in rows if dataset == "turkish"))
        self.assertTrue(all("emotion" not in condition for _, condition, _, _ in rows))

    def test_followup_stage_counts_and_identities(self):
        base = Path("configs/features/optuna_raw_matrix.yaml")
        stage1 = followup.build_manifest(
            stage="stage1",
            base_matrix=base,
            results_root=Path("unused"),
            stability_summary=None,
        )
        self.assertEqual(stage1["expected_jobs"], 33)
        self.assertTrue(
            all(job["experiment_id"] == followup.STAGE1_ID for job in stage1["jobs"])
        )
        self.assertTrue(all(job["target_trials"] == 150 for job in stage1["jobs"]))
        pilot = followup.build_manifest(
            stage="pilot",
            base_matrix=base,
            results_root=Path("unused"),
            stability_summary=None,
        )
        self.assertEqual(pilot["expected_jobs"], 22)
        self.assertEqual({job["inner_seed"] for job in pilot["jobs"]}, {7, 2024})
        self.assertEqual(
            {(job["dataset"], job["condition"]) for job in pilot["jobs"]},
            followup.PILOT_CONDITIONS,
        )

    def test_expansion_requires_gate_and_contains_only_remaining_conditions(self):
        base = Path("configs/features/optuna_raw_matrix.yaml")
        with tempfile.TemporaryDirectory() as directory:
            summary = Path(directory) / "summary.json"
            payload = {
                "source_experiment_ids": [
                    followup.STAGE1_ID,
                    followup.SEED_IDS[7],
                    followup.SEED_IDS[2024],
                ],
                "gate_threshold": 0.03,
                "observed_max_primary_range": 0.04,
                "expand_all": True,
            }
            save_json(payload, summary)
            expansion = followup.build_manifest(
                stage="expansion",
                base_matrix=base,
                results_root=Path("unused"),
                stability_summary=summary,
            )
            self.assertEqual(expansion["expected_jobs"], 44)
            self.assertTrue(
                all(
                    (job["dataset"], job["condition"]) not in followup.PILOT_CONDITIONS
                    for job in expansion["jobs"]
                )
            )
            payload["expand_all"] = False
            save_json(payload, summary)
            with self.assertRaisesRegex(ValueError, "did not trigger"):
                followup.build_manifest(
                    stage="expansion",
                    base_matrix=base,
                    results_root=Path("unused"),
                    stability_summary=summary,
                )

    def test_depth8_selects_every_fold_without_reading_outer_metrics(self):
        base = Path("configs/features/optuna_raw_matrix.yaml")
        jobs = followup._base_jobs(base)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for job in jobs:
                result_dir = followup._result_dir(root, job, followup.STAGE1_ID)
                result_dir.mkdir(parents=True)
                is_boundary = (
                    job["dataset"] == "cmdc"
                    and job["condition"] == "audio_text"
                    and job["fold"] == 2
                )
                save_json(
                    {
                        "completed_trial_count": 150,
                        "best_trial_number": 149 if is_boundary else 10,
                        "suggested_params": {"max_depth": 6 if is_boundary else 3},
                    },
                    result_dir / "best_params.json",
                )
                save_json(
                    {
                        "experiment_id": followup.STAGE1_ID,
                        "completed_trials": 150,
                    },
                    result_dir / "classifier_metadata.json",
                )
            manifest = followup.build_manifest(
                stage="depth8",
                base_matrix=base,
                results_root=root,
                stability_summary=None,
            )
            self.assertEqual(manifest["expected_jobs"], 5)
            self.assertEqual(
                {(job["dataset"], job["condition"]) for job in manifest["jobs"]},
                {("cmdc", "audio_text")},
            )
            self.assertTrue(all(job["search_profile"] == "depth8" for job in manifest["jobs"]))

    def test_stability_gate_uses_pooled_primary_metric_range(self):
        rows = []
        values = {
            ("daic", "text_only"): [0.70, 0.72, 0.71],
            ("cmdc", "audio_text"): [0.60, 0.64, 0.61],
            ("cmdc", "text_only"): [0.10, 0.80, 0.20],
            ("turkish", "text_only"): [0.55, 0.56, 0.57],
        }
        ids = [followup.STAGE1_ID, followup.SEED_IDS[7], followup.SEED_IDS[2024]]
        seeds = [1337, 7, 2024]
        for (dataset, condition), primary_values in values.items():
            for experiment_id, inner_seed, primary_value in zip(ids, seeds, primary_values):
                row = {
                    "dataset": dataset,
                    "condition": condition,
                    "run_name": f"{dataset}_{condition}_run",
                    "experiment_id": experiment_id,
                    "inner_seed": inner_seed,
                    "pooled_positive_f1": 0.5,
                    "pooled_macro_f1": 0.5,
                }
                row[
                    "pooled_macro_f1" if dataset == "turkish" else "pooled_positive_f1"
                ] = primary_value
                rows.append(row)
        with mock.patch.object(stability, "_per_experiment_rows", return_value=rows):
            payload = stability.summarize_stability(Path("unused"), gate_threshold=0.03)
        self.assertTrue(payload["expand_all"])
        self.assertAlmostEqual(payload["observed_max_primary_range"], 0.04)
        self.assertEqual(len(payload["stability_rows"]), 4)
        self.assertEqual(len(payload["pilot_stability_rows"]), 3)
        cmdc = next(
            row
            for row in payload["stability_rows"]
            if row["dataset"] == "cmdc"
        )
        self.assertAlmostEqual(cmdc["primary_range"], 0.04)
        with mock.patch.object(stability, "_per_experiment_rows", return_value=rows):
            stricter = stability.summarize_stability(Path("unused"), gate_threshold=0.05)
        self.assertFalse(stricter["expand_all"])


@unittest.skipIf(
    any(importlib.util.find_spec(name) is None for name in ("optuna", "xgboost", "sklearn", "joblib")),
    "Optuna integration dependencies are not installed",
)
class OptunaIntegrationTests(unittest.TestCase):
    def test_completed_restart_does_not_reload_or_reevaluate_final_partition(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache = root / "cache"
            output = (
                root
                / "hidden_classifiers"
                / "synthetic"
                / "audio_text"
                / "synthetic_run"
                / "fold_0"
                / "xgb_optuna_raw"
            )
            cache.mkdir()
            rng = np.random.default_rng(1337)
            train_rows = []
            train_vectors = []
            for subject_index in range(12):
                label = subject_index % 2
                for sample_index in range(2):
                    train_rows.append(
                        {
                            "sample_id": f"tr-{subject_index}-{sample_index}",
                            "subject_id": f"tr-{subject_index}",
                            "label": label,
                        }
                    )
                    vector = rng.normal(size=4)
                    vector[0] += 2.0 if label else -2.0
                    train_vectors.append(vector)
            final_rows = [
                {"sample_id": "te-0", "subject_id": "te-0", "label": 0},
                {"sample_id": "te-1", "subject_id": "te-1", "label": 1},
            ]
            final_vectors = np.asarray([[-2.0, 0.0, 0.0, 0.0], [2.0, 0.0, 0.0, 0.0]])
            np.savez_compressed(cache / "outer_train.npz", vectors=np.asarray(train_vectors))
            np.savez_compressed(cache / "final_eval.npz", vectors=final_vectors)
            write_jsonl(train_rows, cache / "outer_train_rows.jsonl")
            write_jsonl(final_rows, cache / "final_eval_rows.jsonl")
            save_json(
                {
                    "dataset": "synthetic",
                    "input_modality": "audio_text",
                    "condition": "audio_text",
                    "fold": 0,
                },
                cache / "extraction_metadata.json",
            )
            first_calls = []
            original_loader = optuna_xgb._load_partition

            def recording_loader(cache_dir, name):
                first_calls.append(name)
                return original_loader(cache_dir, name)

            with mock.patch.object(optuna_xgb, "_load_partition", side_effect=recording_loader):
                optuna_xgb.run_optuna_raw_xgb(
                    cache_dir=cache,
                    output_dir=output,
                    objective_name="positive_f1",
                    target_trials=1,
                    inner_folds=3,
                    seed=1337,
                    xgb_threads=1,
                )
            self.assertEqual(first_calls, ["outer_train", "final_eval"])

            restart_calls = []

            def restart_loader(cache_dir, name):
                restart_calls.append(name)
                return original_loader(cache_dir, name)

            with mock.patch.object(optuna_xgb, "_load_partition", side_effect=restart_loader):
                optuna_xgb.run_optuna_raw_xgb(
                    cache_dir=cache,
                    output_dir=output,
                    objective_name="positive_f1",
                    target_trials=1,
                    inner_folds=3,
                    seed=1337,
                    xgb_threads=1,
                )
            self.assertEqual(restart_calls, ["outer_train"])
            self.assertEqual(read_json(output / "classifier_metadata.json")["completed_trials"], 1)


if __name__ == "__main__":
    unittest.main()
