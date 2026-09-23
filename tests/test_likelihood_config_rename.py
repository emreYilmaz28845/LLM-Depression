from pathlib import Path

from src.utils import load_yaml


ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "configs/main"
MERGED = ROOT / "configs/experiments/merged"
ARCHIVE = ROOT / "configs/archive/pre_likelihood_20260917"
PRE_DEFAULT_BACKBONE_ARCHIVE = ROOT / "configs/archive/pre_default_backbone_20260923"
RENAME_MAP = ROOT / "experiments/definitions/likelihood_rename_map.yaml"

MAIN_DIFF_KEYS = {
    "recipe_id",
    "evaluation.sample_prediction_mode",
    "evaluation.headline_mode",
    "output_dirs.run_root",
}
EDAIC = (
    "edaic_audio_only_selposf1_tf.yaml",
    "edaic_audio_text_selposf1_tf.yaml",
    "edaic_text_only_selposf1_tf.yaml",
)


def _flatten(obj, prefix=""):
    if isinstance(obj, dict):
        out = {}
        for key, value in obj.items():
            out.update(_flatten(value, f"{prefix}.{key}" if prefix else str(key)))
        return out
    if isinstance(obj, list):
        out = {}
        for index, value in enumerate(obj):
            out.update(_flatten(value, f"{prefix}[{index}]"))
        return out
    return {prefix: obj}


def _diff_keys(old, new):
    old_flat = _flatten(old)
    new_flat = _flatten(new)
    return {
        key
        for key in set(old_flat) | set(new_flat)
        if old_flat.get(key, "<missing>") != new_flat.get(key, "<missing>")
    }


def _tf_source(old_name: str) -> Path:
    archived = ARCHIVE / "main" / old_name
    if archived.exists():
        return archived
    return MAIN / old_name


def likelihood_main_configs():
    # The 2026-09-17 rename covered the family that existed then. Backbone and
    # prompt variants added later (the Qwen3.8 text-only config, the
    # prompt-context family, the Qwen3-Omni pilot configs) carry their own
    # provenance and are not part of the rename map.
    current = sorted(
        path
        for path in MAIN.glob("*_likelihood_v1*.yaml")
        if "_qwen38_27b" not in path.name
        and "_qwen3omni_30b_a3b" not in path.name
        and "_promptcontext_v1" not in path.name
    )
    return [
        (PRE_DEFAULT_BACKBONE_ARCHIVE / path.name)
        if (PRE_DEFAULT_BACKBONE_ARCHIVE / path.name).is_file()
        else path
        for path in current
    ]


def test_every_main_likelihood_config_matches_its_tf_source_exactly() -> None:
    paths = likelihood_main_configs()
    assert len(paths) == 82
    for new_path in paths:
        assert "_tf" not in new_path.name
        assert new_path.name.count("_likelihood_v1") == 1
        old_name = new_path.name.replace("_likelihood_v1", "_tf")
        old_path = _tf_source(old_name)
        assert old_path.exists(), f"missing tf source for {new_path.name}"
        old = load_yaml(old_path)
        new = load_yaml(new_path)
        assert _diff_keys(old, new) == MAIN_DIFF_KEYS, new_path.name
        assert new["recipe_id"] == old["recipe_id"].replace(
            "_selmacrof1_tf", "_selmacrof1_likelihood"
        )
        assert new["evaluation"]["sample_prediction_mode"] == "likelihood"
        assert new["evaluation"]["headline_mode"] == "likelihood"
        assert new["output_dirs"]["run_root"] != old["output_dirs"]["run_root"]
        assert "_likelihood" in new["output_dirs"]["run_root"]


def test_merged_likelihood_configs_repoint_components_only() -> None:
    paths = sorted(MERGED.glob("*_likelihood_v1.yaml"))
    assert len(paths) == 6
    for new_path in paths:
        old_name = new_path.name.replace("_likelihood_v1.yaml", ".yaml")
        old_path = ARCHIVE / "merged" / old_name
        if not old_path.exists():
            old_path = MERGED / old_name
        assert old_path.exists(), f"missing merged tf source for {new_path.name}"
        old = load_yaml(old_path)
        new = load_yaml(new_path)
        expected = {"recipe_id", "output_dirs.run_root"} | {
            f"components[{index}].config" for index in range(len(old["components"]))
        }
        assert _diff_keys(old, new) == expected, new_path.name
        assert new["output_dirs"]["merged_root"] == old["output_dirs"]["merged_root"]
        assert new["recipe_id"] == old["recipe_id"].replace(
            "_selmacrof1_tf", "_selmacrof1_likelihood"
        )
        for component in new["components"]:
            assert (ROOT / component["config"]).exists(), component["config"]


def test_rename_map_is_complete_and_consistent() -> None:
    mapping = load_yaml(RENAME_MAP)
    assert mapping["schema_version"] == "audiollm.likelihood_rename_map.v1"
    assert len(mapping["configs"]) == 82
    assert len(mapping["merged"]) == 6
    for entry in mapping["configs"] + mapping["merged"]:
        assert (ROOT / entry["new_path"]).exists(), entry["new_path"]
        assert (ROOT / entry["old_path"]).exists() or (ROOT / entry["archive_path"]).exists()
        assert entry["new_run_root"] != entry["old_run_root"]
        assert entry["new_recipe_id"].endswith("_likelihood_v1") or "_likelihood_" in entry["new_recipe_id"]
    assert {entry["new_path"] for entry in mapping["configs"]} == {
        f"configs/main/{path.name}" for path in likelihood_main_configs()
    }
    assert {entry["new_path"] for entry in mapping["merged"]} == {
        f"configs/experiments/merged/{path.name}" for path in MERGED.glob("*_likelihood_v1.yaml")
    }


def test_edaic_configs_stay_teacher_forced_and_untouched() -> None:
    for name in EDAIC:
        config = load_yaml(MAIN / name)
        assert config["evaluation"]["sample_prediction_mode"] == "original_teacher_forced"
        assert config["evaluation"]["headline_mode"] == "original_teacher_forced"
    assert not list(MAIN.glob("edaic_*_likelihood_v1*.yaml"))
