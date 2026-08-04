from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.daic_derived_views import materialize_cached_hidden_views


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--views", required=True)
    args = parser.parse_args()
    materialize_cached_hidden_views(
        args.cache_root, [view for view in args.views.split(",") if view]
    )


if __name__ == "__main__":
    main()
