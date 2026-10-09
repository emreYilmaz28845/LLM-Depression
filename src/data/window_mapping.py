"""Reference mapping from subdivided windows to the canonical segmentation.

D3TEC and Androids manifests were originally built with equal-duration windows
of 30 seconds, and their segment transcript files are keyed by those canonical
window ids. The window15 comparison arm subdivides the same responses at 15
seconds.

``ceil(duration/15)`` windows are NOT in general a subdivision of
``ceil(duration/30)`` windows: for duration 70 the canonical widths are 70/3
and a 15-second child window can cross a canonical boundary. The mapping is
therefore a *reference*: each child window references the canonical window
whose interval contains the child's start. It makes no containment or alignment
claim, and the canonical segment transcript is recorded as a reference, never
as the child window's aligned text. The model input text is unaffected: the
harmonized runtime uses the full subject transcript for these datasets.
"""

from __future__ import annotations

import math


def source_reference_index(start_time: float, duration: float, source_seconds: float = 30.0) -> int:
    """Canonical window index referenced by a subdivided window's start.

    Both segmentations are contiguous equal-duration windows over
    ``[0, duration)``. The returned index is the canonical window containing
    ``start_time``; a child window may extend past that window's end, which is
    why callers must treat the result as a reference rather than containment.
    """

    if duration <= 0 or source_seconds <= 0:
        raise ValueError("duration and source_seconds must be positive")
    count = max(1, int(math.ceil(duration / source_seconds)))
    width = duration / count
    index = int(math.floor((float(start_time) + 1e-9) / width))
    return min(count - 1, max(0, index))


def reference_fields(
    segment_seconds: float,
    canonical_seconds: float,
    start_time: float,
    duration: float,
) -> dict:
    """Reference fields for a discovery row.

    Returns an empty dict on the canonical segmentation so 30-second manifests
    stay byte-identical to the historical schema; only the subdivided path adds
    the reference index.
    """

    if segment_seconds == canonical_seconds:
        return {}
    return {"source_reference_index": source_reference_index(start_time, duration, canonical_seconds)}
