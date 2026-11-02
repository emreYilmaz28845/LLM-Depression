#!/usr/bin/env python3
"""Simple acoustic / channel statistics per subject, as a non-neural baseline for the dataset probe.

For every audio unit in the given manifests (a window with start/end times, a
whole file, or DAIC participant spans) the signal is read, mixed to mono,
resampled to 16 kHz, and framed (25 ms Hann, 10 ms hop). Per unit:

- duration, active-frame ratio (frames within 30 dB of the unit's 95th
  percentile level), level percentiles (10/50/90) and their range, clipping
  ratio and DC offset (on the original signal);
- over active frames, mean and std of spectral centroid, bandwidth, 85 %
  roll-off, flatness, slope (dB per kHz, 0-8 kHz) and zero-crossing rate;
- relative long-term spectrum in 8 bands for active frames and for the
  quietest 10 % of frames (a noise-floor / channel response estimate);
- effective bandwidth (highest frequency within 50 dB of the active
  long-term spectrum peak);
- mean and std of 13 MFCCs over active frames.

A subject's vector is the mean over its units. The native sample rate is
recorded per unit but is not a feature. No transcript, label-derived or
participant-identifying field is written except subject id, dataset and label.
"""

from __future__ import annotations

import argparse
import ast
import json
import subprocess
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.fft import dct
from scipy.signal import resample_poly

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SR = 16000
FRAME, HOP, NFFT = 400, 160, 512
BANDS = ((0, 250), (250, 500), (500, 1000), (1000, 2000), (2000, 3000), (3000, 4000), (4000, 6000), (6000, 8000))
FREQS = np.fft.rfftfreq(NFFT, 1.0 / SR)
GPFS_PREFIX = "/gpfs/projects/etur92/ozu647717/AudioLLM/Datasets/"
LOCAL_PREFIX = "/media/emre/Backup/AudioLLM/Datasets/"


def _mel_filterbank(n_mels: int = 40) -> np.ndarray:
    mel = lambda f: 2595.0 * np.log10(1.0 + f / 700.0)  # noqa: E731
    inv = lambda m: 700.0 * (10 ** (m / 2595.0) - 1.0)  # noqa: E731
    points = inv(np.linspace(mel(0.0), mel(SR / 2), n_mels + 2))
    bank = np.zeros((n_mels, len(FREQS)))
    for i in range(n_mels):
        lo, mid, hi = points[i], points[i + 1], points[i + 2]
        bank[i] = np.clip(np.minimum((FREQS - lo) / (mid - lo), (hi - FREQS) / (hi - mid)), 0.0, None)
    return bank


MEL = _mel_filterbank()


def feature_names() -> list[str]:
    names = ["duration_s", "active_ratio", "level_p10_db", "level_p50_db", "level_p90_db", "level_range_db",
             "clip_ratio", "dc_offset"]
    for f in ("centroid_hz", "bandwidth_hz", "rolloff85_hz", "flatness", "slope_db_per_khz", "zcr"):
        names += [f"{f}_mean", f"{f}_std"]
    names += [f"ltas_active_{lo}_{hi}" for lo, hi in BANDS] + [f"ltas_quiet_{lo}_{hi}" for lo, hi in BANDS]
    names += ["effective_bandwidth_hz"]
    names += [f"mfcc{i}_mean" for i in range(13)] + [f"mfcc{i}_std" for i in range(13)]
    return names


def _band_profile(power: np.ndarray) -> np.ndarray:
    total = power.sum() + 1e-20
    return np.asarray([10 * np.log10(power[(FREQS >= lo) & (FREQS < hi)].sum() / total + 1e-12) for lo, hi in BANDS])


def unit_features(x: np.ndarray, native_sr: int) -> np.ndarray:
    clip_ratio = float(np.mean(np.abs(x) >= 0.99))
    dc = float(abs(np.mean(x)))
    if native_sr != SR:
        x = resample_poly(x, SR, native_sr)
    if len(x) < FRAME:
        x = np.pad(x, (0, FRAME - len(x)))
    n = 1 + (len(x) - FRAME) // HOP
    frames = np.lib.stride_tricks.as_strided(x, (n, FRAME), (x.strides[0] * HOP, x.strides[0])).copy()
    level = 10 * np.log10(np.mean(frames ** 2, axis=1) + 1e-12)
    spec = np.abs(np.fft.rfft(frames * np.hanning(FRAME), NFFT, axis=1)) ** 2
    active = level > np.percentile(level, 95) - 30.0
    if active.sum() < 3:
        active = np.ones(n, dtype=bool)
    quiet = level <= np.percentile(level, 10)
    sa = spec[active]
    norm = sa.sum(1, keepdims=True) + 1e-20
    centroid = (sa * FREQS).sum(1) / norm[:, 0]
    bandwidth = np.sqrt((sa * (FREQS[None] - centroid[:, None]) ** 2).sum(1) / norm[:, 0])
    rolloff = FREQS[np.argmax(np.cumsum(sa, 1) >= 0.85 * norm, axis=1)]
    flatness = np.exp(np.mean(np.log(sa + 1e-20), 1)) / (np.mean(sa, 1) + 1e-20)
    slope = np.polyfit(FREQS / 1000.0, (10 * np.log10(sa + 1e-20)).T, 1)[0]
    zcr = np.mean(np.abs(np.diff(np.sign(frames[active]), axis=1)) > 0, axis=1)
    ltas_active, ltas_quiet = sa.mean(0), spec[quiet].mean(0)
    ltas_db = 10 * np.log10(ltas_active + 1e-20)
    above = np.nonzero(ltas_db >= ltas_db.max() - 50.0)[0]
    mfcc = dct(np.log(sa @ MEL.T + 1e-10), type=2, norm="ortho", axis=1)[:, :13]
    p10, p50, p90 = np.percentile(level, [10, 50, 90])
    out = [len(x) / SR, float(active.mean()), p10, p50, p90, p90 - p10, clip_ratio, dc]
    for v in (centroid, bandwidth, rolloff, flatness, slope, zcr):
        out += [float(np.mean(v)), float(np.std(v))]
    out += _band_profile(ltas_active).tolist() + _band_profile(ltas_quiet).tolist()
    out += [float(FREQS[above.max()])]
    out += mfcc.mean(0).tolist() + mfcc.std(0).tolist()
    return np.asarray(out, dtype=np.float64)


def _local(path: str, path_map: dict[str, str]) -> str:
    if path in path_map:
        return path_map[path]
    return path.replace(GPFS_PREFIX, LOCAL_PREFIX, 1) if path.startswith(GPFS_PREFIX) else path


def _read(path: str, start: float | None = None, end: float | None = None,
          spans: list[tuple[int, int]] | None = None) -> tuple[np.ndarray, int]:
    info = sf.info(path)
    if spans:
        parts = [sf.read(path, start=a, stop=b, dtype="float64", always_2d=True)[0] for a, b in spans]
        x = np.concatenate(parts)
    elif start is not None and end is not None and end > start:
        x = sf.read(path, start=int(round(start * info.samplerate)), stop=int(round(end * info.samplerate)),
                    dtype="float64", always_2d=True)[0]
    else:
        x = sf.read(path, dtype="float64", always_2d=True)[0]
    return x.mean(1), int(info.samplerate)


def units_from_manifest(dataset: str, manifest: Path, path_map: dict[str, str]) -> list[dict]:
    """One entry per distinct audio unit, keyed by (path, start, end) or DAIC sample."""
    seen, units = set(), []
    for line in manifest.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        path = _local(str(r["audio_path"]), path_map)
        spans = None
        if r.get("audio_spans"):
            raw = r["audio_spans"]
            raw = ast.literal_eval(raw) if isinstance(raw, str) else raw
            spans = [(int(s["start_frame"]), int(s["end_frame"])) for s in raw]
            key = (path, r["sample_id"])
            start = end = None
        else:
            start = float(r["start_time"]) if str(r.get("start_time", "")).strip() else None
            end = float(r["end_time"]) if str(r.get("end_time", "")).strip() else None
            key = (path, start, end)
        if key in seen:
            continue
        seen.add(key)
        units.append({"dataset": dataset, "subject_id": f"{dataset}::{r['subject_id']}", "label": int(r["label"]),
                      "path": path, "start": start, "end": end, "spans": spans})
    return units


def _work(unit: dict) -> tuple[np.ndarray | None, int | None, str | None]:
    try:
        x, sr = _read(unit["path"], unit["start"], unit["end"], unit["spans"])
        return unit_features(x, sr), sr, None
    except Exception as error:  # recorded and reported, never silently dropped
        return None, None, f"{unit['path']}: {type(error).__name__}: {error}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest", action="append", required=True, help="dataset=path/to/manifest.jsonl")
    parser.add_argument("--path-map", type=Path, help="JSON map from manifest audio paths to local paths.")
    parser.add_argument("--output", required=True, type=Path, help="Output .npz (a .json sidecar is written next to it).")
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"Refusing to overwrite {args.output}.")
    path_map = json.loads(args.path_map.read_text()) if args.path_map else {}
    units = []
    for item in args.manifest:
        dataset, path = item.split("=", 1)
        units += units_from_manifest(dataset, Path(path), path_map)
    print(f"{len(units)} units", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(_work, units, chunksize=16))
    errors = [e for _, _, e in results if e]
    if errors:
        raise SystemExit(f"{len(errors)} units failed, e.g. {errors[:3]}")
    by_subject: dict[str, list[np.ndarray]] = defaultdict(list)
    meta: dict[str, tuple[str, int]] = {}
    rates: dict[str, Counter] = defaultdict(Counter)
    for unit, (vector, sr, _) in zip(units, results):
        by_subject[unit["subject_id"]].append(vector)
        meta[unit["subject_id"]] = (unit["dataset"], unit["label"])
        rates[unit["dataset"]][sr] += 1
    subjects = sorted(by_subject)
    x = np.stack([np.mean(by_subject[s], axis=0) for s in subjects])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, x=x, subject_id=np.asarray(subjects), dataset=np.asarray([meta[s][0] for s in subjects]),
             label=np.asarray([meta[s][1] for s in subjects]), feature_names=np.asarray(feature_names()))
    sidecar = {"git_head": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                                          cwd=PROJECT_ROOT).stdout.strip(),
               "manifests": args.manifest, "units": len(units), "subjects": len(subjects),
               "units_per_dataset": dict(Counter(u["dataset"] for u in units)),
               "native_sample_rates_per_dataset": {d: {str(k): v for k, v in c.items()} for d, c in rates.items()},
               "feature_names": feature_names()}
    args.output.with_suffix(".json").write_text(json.dumps(sidecar, indent=2))
    print(json.dumps({k: v for k, v in sidecar.items() if k != "feature_names"}, indent=2))


if __name__ == "__main__":
    main()
