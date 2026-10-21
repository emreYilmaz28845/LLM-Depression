#!/usr/bin/env python3
"""Draw token-level attention maps saved by probe_audio_recording_shortcut.py --attention-only.

One figure per saved window: the decision position's attention (head mean) to
every prompt token, layer by layer, on a log colour scale, with a bar under it
that marks which prompt region each token belongs to.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

REGION_COLORS = {
    "sink": "#7f7f7f",
    "system": "#1f77b4",
    "audio_marker": "#ff7f0e",
    "audio": "#d62728",
    "audio_nonspeech": "#ff9896",
    "user_text": "#2ca02c",
    "transcript": "#9467bd",
    "template": "#8c564b",
    "label": "#e377c2",
}
REGION_LABELS = {
    "sink": "first token",
    "system": "system prompt",
    "audio_marker": "audio start/end",
    "audio": "audio (speech)",
    "audio_nonspeech": "audio (non-speech)",
    "user_text": "instruction",
    "transcript": "transcript",
    "template": "chat template",
    "label": "label",
}


def plot(path: Path, out_dir: Path, variant: str = "fine-tuned", dataset: str = "Androids") -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, LogNorm
    from matplotlib.patches import Patch

    data = np.load(path)
    weights = data["weights"]  # [layers, keys]
    regions = data["regions"][: weights.shape[1]]
    names = [str(name) for name in data["region_names"]]
    label = "Depressed" if int(data["label"]) == 1 else "Non-depressed"
    model = "audio+text" if path.name.startswith("audio_text") else "audio-only"

    fig, (ax_map, ax_bar) = plt.subplots(
        2, 1, figsize=(16, 7), sharex=True, gridspec_kw={"height_ratios": [12, 1], "hspace": 0.05}
    )
    floor = 1e-6
    image = ax_map.imshow(
        np.clip(weights, floor, 1.0),
        aspect="auto",
        cmap="inferno",
        norm=LogNorm(vmin=floor, vmax=1.0),
        interpolation="nearest",
        origin="lower",
    )
    ax_map.set_ylabel("layer")
    ax_map.set_title(f"Qwen3-Omni {model} ({variant}), {dataset} fold 0, true label {label}: "
                     "attention of the answer position to each prompt token")
    colorbar = fig.colorbar(image, ax=[ax_map, ax_bar], pad=0.01, fraction=0.02)
    colorbar.set_label("attention weight (head mean, log scale)")

    used = [names[code] for code in sorted(set(regions.tolist()))]
    cmap = ListedColormap([REGION_COLORS.get(name, "#000000") for name in names])
    ax_bar.imshow(regions[None, :], aspect="auto", cmap=cmap, vmin=-0.5, vmax=len(names) - 0.5,
                  interpolation="nearest")
    ax_bar.set_yticks([])
    ax_bar.set_xlabel("prompt token position")
    ax_bar.legend(
        handles=[Patch(color=REGION_COLORS.get(n, "#000000"), label=REGION_LABELS.get(n, n)) for n in used],
        loc="upper center", bbox_to_anchor=(0.5, -1.6), ncol=len(used), frameon=False,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"{path.stem}.png"
    fig.savefig(target, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--dataset", default="Androids", help="dataset name for the title")
    parser.add_argument("--variant", default="fine-tuned", help="title tag, e.g. zero-shot")
    args = parser.parse_args()
    for path in args.inputs:
        print(plot(path, args.out_dir, args.variant, args.dataset))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
