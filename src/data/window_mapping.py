"""Map subdivided audio windows back to their canonical source segmentation.

D3TEC and Androids manifests were originally built with equal-duration windows
of 30 seconds, and their segment transcripts are keyed by those canonical
window ids. The window15 comparison arm subdivides the same responses at 15
seconds, so each treatment window must resolve the transcript of the canonical
30-second window that contains it. This helper computes that containing index;
both segmentations are contiguous equal-duration windows over ``[0, duration)``.
"""

from __future__ import annotations

import math


def source_window_index(start_time: float, duration: float, source_seconds: float = 30.0) -> int:
    """Index of the canonical ``source_seconds`` window containing ``start_time``."""

    if duration <= 0 or source_seconds <= 0:
        raise ValueError("duration and source_seconds must be positive")
    count = max(1, int(math.ceil(duration / source_seconds)))
    width = duration / count
    index = int(math.floor((float(start_time) + 1e-9) / width))
    return min(count - 1, max(0, index))
