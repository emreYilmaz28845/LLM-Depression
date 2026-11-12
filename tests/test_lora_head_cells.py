from pathlib import Path

import numpy as np

from scripts import lora_head_cells as L
from scripts import overlap_oracle_analysis as O


def test_route_dir_per_seed_and_modality():
    r = Path("/x")
    assert L.route_dir(r, "turkish", "text_only", 1337) == r / "heads_lora_textrerun/turkish/text_only_native"
    assert L.route_dir(r, "turkish", "audio_text", 1337) == r / "heads_lora/turkish/audio_text_native"
    assert L.route_dir(r, "daic", "audio_only", 7) == r / "heads_lora_seeds/daic/audio_only_native/s7"


def test_seed_of_checkpoint():
    assert L.seed_of_checkpoint("/a/q3ms_d3tec_audio_text_native_s2024_f3_r1/fold_3/best_model") == 2024
    assert L.seed_of_checkpoint("/a/qwen3omni_d3tec_audio_text_f0_prod/fold_0/best_model") is None


def test_oracle_gain_zero_for_identical_models():
    y = np.array([0, 1, 0, 1]); p = np.array([0, 1, 1, 1]); fold = np.zeros(4, int)
    gain, dr = O.oracle_gain(y, p, p, fold)
    assert gain == 0 and dr == 0


def test_oracle_gain_positive_when_errors_differ():
    y = np.array([0, 1, 0, 1]); a = np.array([1, 1, 0, 1]); b = np.array([0, 1, 0, 0]); fold = np.zeros(4, int)
    gain, dr = O.oracle_gain(y, a, b, fold)
    assert gain > 0 and dr == .5
