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


def _csv_int_list(value: str | None) -> list[int] | None:
    if value is None:
        return None
    return [int(item) for item in value.split(",") if item.strip()]


def _csv_str_list(value: str | None) -> list[str] | None:
    if value is None:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate deterministic experiment-group report (report.json + report.md)."
    )
    parser.add_argument(
        "--db",
        default=str(PROJECT_ROOT / registry.DEFAULT_DB_PATH),
        help="SQLite registry path",
    )
    parser.add_argument("--attempts", required=True, help="Comma-separated attempt IDs")
    parser.add_argument("--metric-name", required=True)
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--backend", required=True)
    parser.add_argument("--view", required=True)
    parser.add_argument("--aggregation", required=True)
    parser.add_argument("--compare-a", default=None, help="Baseline attempt IDs for paired deltas")
    parser.add_argument("--compare-b", default=None, help="Treatment attempt IDs for paired deltas")
    parser.add_argument("--expected-seeds", default=None, help="Comma-separated integers")
    parser.add_argument("--expected-folds", default=None, help="Comma-separated integers")
    parser.add_argument("--research-question", default=None)
    parser.add_argument("--hypothesis", default=None)
    parser.add_argument("--baseline", default=None)
    parser.add_argument("--treatment", default=None)
    parser.add_argument("--output", default="outputs/experiment_reports", help="Output directory")
    parser.add_argument("--with-timestamp", action="store_true", help="Include generation timestamp (default: deterministic output)")
    parser.add_argument("--conclusion", default=None, help="Researcher-authored interpretation")
    args = parser.parse_args()

    attempt_ids = _csv_str_list(args.attempts)
    if not attempt_ids:
        parser.error("--attempts must list at least one attempt id")
    connection = registry.connect(args.db)
    try:
        generated_at = format_utc_timestamp(utc_now()) if args.with_timestamp else None
        payload = reporting.build_group_report(
            connection,
            attempt_ids,
            metric_name=args.metric_name,
            namespace=args.namespace,
            backend=args.backend,
            view=args.view,
            aggregation=args.aggregation,
            compare_a=_csv_str_list(args.compare_a),
            compare_b=_csv_str_list(args.compare_b),
            expected_seeds=_csv_int_list(args.expected_seeds),
            expected_folds=_csv_int_list(args.expected_folds),
            research_question=args.research_question,
            hypothesis=args.hypothesis,
            baseline=args.baseline,
            treatment=args.treatment,
            generated_at_utc=generated_at,
            conclusion=args.conclusion,
        )
    finally:
        connection.close()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "group_report.json"
    md_path = output_dir / "group_report.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    md_path.write_text(reporting.render_group_report_markdown(payload), encoding="utf-8")
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")
    if not payload["compatibility"]["ok"]:
        for issue in payload["compatibility"]["issues"]:
            print(f"compatibility issue: {issue}")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
