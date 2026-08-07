from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.experiment_tracking import registry, reporting
from src.experiment_tracking.canonical import format_utc_timestamp, utc_now


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate deterministic run report (report.json + report.md) from registry records."
    )
    parser.add_argument(
        "--db",
        default=str(PROJECT_ROOT / registry.DEFAULT_DB_PATH),
        help="SQLite registry path",
    )
    parser.add_argument("--attempt-id", required=True)
    parser.add_argument("--fold", type=int, default=None)
    parser.add_argument("--output", default="outputs/experiment_reports", help="Output directory")
    parser.add_argument("--with-timestamp", action="store_true", help="Include generation timestamp (default: deterministic output)")
    parser.add_argument("--conclusion", default=None, help="Researcher-authored interpretation")
    args = parser.parse_args()

    connection = registry.connect(args.db)
    try:
        generated_at = format_utc_timestamp(utc_now()) if args.with_timestamp else None
        payload = reporting.build_run_report(
            connection,
            args.attempt_id,
            fold=args.fold,
            generated_at_utc=generated_at,
            conclusion=args.conclusion,
        )
    finally:
        connection.close()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "report.json"
    md_path = output_dir / "report.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    md_path.write_text(reporting.render_run_report_markdown(payload), encoding="utf-8")
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
