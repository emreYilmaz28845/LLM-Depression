import numpy as np
import pytest
import torch

from scripts.probe_audio_attention import (
    REGION_INDEX,
    REGIONS,
    assign_regions,
    audio_run_lengths,
    causal_rows_softmax,
    expand_audio_placeholders,
    parse_judgment,
    region_mass,
    uniform_share,
)

AUDIO, START, END = 9, 7, 8


def test_expand_audio_placeholders_matches_runs():
    ids = [1, START, AUDIO, AUDIO, AUDIO, END, 2]
    assert audio_run_lengths(ids, AUDIO) == [3]
    text = "a<|audio_start|><|audio_pad|><|audio_end|>b"
    assert expand_audio_placeholders(text, [3]).count("<|audio_pad|>") == 3
    with pytest.raises(ValueError):
        expand_audio_placeholders(text, [3, 2])


def test_assign_regions_with_offsets():
    transcript = "hello there"
    text = (
        "<|im_start|>system\nS<|im_end|>\n<|im_start|>user\nA<|audio_pad|>\n"
        f"The transcript of the subject's speech is:\n{transcript}\n\nQ<|im_end|>\n"
        "<|im_start|>assistant\nDepressed"
    )
    # One fake token per landmark, with the character start of each landmark.
    landmarks = [
        (0, 1),  # sink
        (text.find("S<|im_end|>"), AUDIO + 100),  # system text
        (text.find("A<|audio_pad|>"), 101),  # user text
        (text.find("<|audio_pad|>"), AUDIO),  # audio
        (text.find(transcript), 102),  # transcript
        (text.find("Q<|im_end|>"), 103),  # user text
        (text.rfind("<|im_start|>assistant"), 104),  # template
        (text.rfind("Depressed"), 105),  # label
    ]
    offsets = [(start, start + 1) for start, _ in landmarks]
    ids = [token for _, token in landmarks]
    regions, method = assign_regions(
        input_ids=ids,
        offsets=offsets,
        text=text,
        prompt_len=7,
        transcript=transcript,
        audio_token_id=AUDIO,
        audio_marker_ids={START, END},
    )
    assert method == "char_offsets"
    assert [REGIONS[code] for code in regions] == [
        "sink",
        "system",
        "user_text",
        "audio",
        "transcript",
        "user_text",
        "template",
        "label",
    ]


def test_assign_regions_ids_only():
    regions, method = assign_regions(
        input_ids=[1, START, AUDIO, END, 2, 3],
        offsets=None,
        text="",
        prompt_len=5,
        transcript="",
        audio_token_id=AUDIO,
        audio_marker_ids={START, END},
    )
    assert method == "ids_only"
    assert [REGIONS[code] for code in regions] == [
        "sink",
        "audio_marker",
        "audio",
        "audio_marker",
        "user_text",
        "label",
    ]


def test_region_mass_and_uniform_share():
    regions = np.array([REGION_INDEX["sink"], REGION_INDEX["audio"], REGION_INDEX["audio"], REGION_INDEX["label"]])
    weights = np.array([[[0.5, 0.2, 0.3, 0.0]]])  # [heads=1, queries=1, keys=4]
    mass = region_mass(weights, regions)
    assert mass.shape == (1, 1, len(REGIONS))
    assert mass[0, 0, REGION_INDEX["audio"]] == pytest.approx(0.5)
    assert mass[0, 0].sum() == pytest.approx(1.0)
    share = uniform_share(regions, query_position=2)
    assert share[REGION_INDEX["audio"]] == pytest.approx(2 / 3)
    assert share[REGION_INDEX["label"]] == 0


def test_causal_rows_softmax_matches_full_causal_attention():
    torch.manual_seed(0)
    scores = torch.randn(1, 2, 5, 5)
    mask = torch.triu(torch.ones(5, 5, dtype=torch.bool), diagonal=1)
    full = torch.softmax(scores.masked_fill(mask, float("-inf")), dim=-1)
    rows = [2, 4]
    recomputed = causal_rows_softmax(scores[:, :, rows], rows)
    assert torch.allclose(recomputed, full[:, :, rows], atol=1e-6)


def test_parse_judgment():
    assert parse_judgment("Speech rate: slow\nJudgment: Non-depressed") == "Non-depressed"
    assert parse_judgment("...\nJudgment: Depressed") == "Depressed"
    assert parse_judgment("no verdict") == "INVALID"
