#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import ensure_dir  # noqa: E402


DEFAULT_ROOT = Path("/media/emre/Backup/AudioLLM/Datasets/Turkish")
DEFAULT_TRANSCRIPTS = "whisper_transcripts.jsonl"
DEFAULT_REVIEW_JSONL = (
    PROJECT_ROOT
    / "outputs"
    / "reviews"
    / "turkish_transcripts"
    / "turkish_transcript_review.jsonl"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "reviews" / "turkish_transcripts"
DEFAULT_REPORT_MD = PROJECT_ROOT / "TURKISH_TRANSCRIPT_REPAIR_REPORT.md"

ELLIPSIS_ENDINGS = ("...", "…")
TERMINAL_SENTENCE_RE = re.compile(r"([.?!])\s")
SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+([,.;!?])")
MULTISPACE_RE = re.compile(r"\s+")

# High-confidence hard failures from the review pass. These are non-Turkish or
# unrelated insertions and should not be passed downstream as usable text.
FAIL_FILENAMES = {
    "ak3-1-11-ank.wav",
    "cy2-1-9-ank+depr.wav",
    "sb2-1-9-ank+depr.wav",
}

# Exact or near-exact mechanical fixes only. These are intentionally narrow.
TEXT_REPLACEMENTS: list[tuple[str, re.Pattern[str], str]] = [
    (
        "greeting_ck1",
        re.compile(r"^Elhamdülillah yiyeyim,\s*buraya", re.IGNORECASE),
        "Elhamdülillah, iyiyim. Buraya",
    ),
    ("greeting_ey1", re.compile(r"^yiyeyim\.\s*"), "İyiyim. "),
    (
        "greeting_hu1",
        re.compile(r"^Teşekkür ederim hocam, iyi yiyeyim\.\s*"),
        "Teşekkür ederim hocam, iyiyim. ",
    ),
    (
        "greeting_sm1",
        re.compile(r"^Teşekkür ederim, iyi yiyeyim\.\s*"),
        "Teşekkür ederim, iyiyim. ",
    ),
    ("yazaltiyor", re.compile(r"\byazaltıyor\b"), "azaltıyor"),
    ("arara", re.compile(r"\barara\b"), "ara ara"),
    ("talleri", re.compile(r"\btalleri\b"), "tahlilleri"),
    ("gaydir", re.compile(r"\b([0-9]+-[0-9]+) gaydır\b"), r"\1 aydır"),
    ("tiz_izliyorum", re.compile(r"\btiz izliyorum\b"), "dizi izliyorum"),
    ("hersi_gun", re.compile(r"\bhersi gün\b"), "her gün"),
    ("ahraba", re.compile(r"\bahraba\b"), "akraba"),
    ("duyuklarla", re.compile(r"\bduyuklarla\b"), "duygularla"),
    ("kucukdum", re.compile(r"\bküçükdum\b"), "küçüktüm"),
    ("firmizlemeyi", re.compile(r"\bfirmizlemeyi\b"), "film izlemeyi"),
    ("mutluluyorum", re.compile(r"\bmutluluyorum\b"), "mutlu oluyorum"),
    ("farkisehirlerden", re.compile(r"\bFarkışehirlerden\b"), "farklı şehirlerden"),
    ("kiskardeslerimle_u", re.compile(r"\bKıskardeşlerimle\b"), "Kız kardeşlerimle"),
    ("kiskardeslerimle_l", re.compile(r"\bkıskardeşlerimle\b"), "kız kardeşlerimle"),
    ("ozgunum", re.compile(r"\bözgünüm\b"), "üzgünüm"),
    ("esim_gelmisti", re.compile(r"\besim gelmişti\b"), "eşim gelmişti"),
    ("cagiriyorlar_isaret", re.compile(r"\bçağırıyorlar işaret\b"), "çağırıyorlar işte"),
    ("tapama_taktiysam", re.compile(r"\btapama taktıysam\b"), "kafama taktıysam"),
    ("pek_firsat", re.compile(r"\bPeki fırsat\b"), "Pek fırsat"),
    ("tahli_sonuclarim", re.compile(r"\btahli sonuçlarım\b"), "tahlil sonuçlarım"),
    ("sigara_istim", re.compile(r"\bsigara istim\b"), "sigara içtim"),
    ("nelaclarini", re.compile(r"\bnelaçlarını\b"), "ilaçlarını"),
    ("randova", re.compile(r"\brandova\b"), "randevu"),
    ("oygudan", re.compile(r"\bOygudan\b"), "Uykudan"),
    ("kavallimizi", re.compile(r"\bKavallımızı\b"), "Kahvaltımızı"),
    ("kavaltimizi", re.compile(r"\bKavaltımızı\b"), "Kahvaltımızı"),
    ("dorunlarim", re.compile(r"\bDorunlarım\b"), "Torunlarım"),
]

# Only remove fragments that are clearly interviewer prompts. Ambiguous content
# is left in place and surfaced for manual review instead.
PROMPT_REMOVALS: list[tuple[str, re.Pattern[str]]] = [
    ("remove_efendim", re.compile(r"\bEfendim\?\s*", re.IGNORECASE)),
    ("remove_baska_neler", re.compile(r"\bBaşka neler\?\s*", re.IGNORECASE)),
    (
        "remove_bugun_ne_yaptiniz",
        re.compile(r"\bPeki bugün ne yapt[ıi]n[ıi]z buraya gelinceye kadar\?\s*", re.IGNORECASE),
    ),
    ("remove_bir_olay_yasadin", re.compile(r"^Bir olay yaşad[ıi]n mı\?\s*", re.IGNORECASE)),
    (
        "remove_mutlu_derken",
        re.compile(r"\bMutlu derken çok fazla mutlu olmak mı\?\s*", re.IGNORECASE),
    ),
    (
        "remove_son_zamanlarda_mutlu_olay",
        re.compile(r"\bSon zamanlarda siz(?:i)? mutlu eden bir olay yaşadınız mı\?\s*", re.IGNORECASE),
    ),
    (
        "remove_mutlu_eden_bir_olay",
        re.compile(r"\bMutlu eden bir olay yaşadınız mı\?\s*", re.IGNORECASE),
    ),
    (
        "remove_p1y_mutlu_olay",
        re.compile(
            r"^Peki son bir yıllık ya da altı ay içerisinde sizi mutlu eden başka bir olay yaşadınız mı\?\s*",
            re.IGNORECASE,
        ),
    ),
    (
        "remove_gunluk_hayat_prompt",
        re.compile(r"\bPeki, günlük hayatınızda evde meselesi mi\?\s*", re.IGNORECASE),
    ),
    (
        "remove_evde_cocuklarla_prompt",
        re.compile(
            r"^Peki evde çocuklarla ailenizle birlikte neler yapmak hoşunuza girer, meraklı edersiniz\?\s*",
            re.IGNORECASE,
        ),
    ),
    ("remove_iyi_geliyor_musunuz", re.compile(r"\bİyi geliyor musunuz\?\s*", re.IGNORECASE)),
    ("remove_standalone_peki_ee", re.compile(r"\bPeki,\s*ee\.\.\.\s*", re.IGNORECASE)),
]

REMAINING_PROMPT_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("remaining_efendim", re.compile(r"\bEfendim\??", re.IGNORECASE)),
    ("remaining_bugun_ne_yaptiniz", re.compile(r"bugün ne yapt[ıi]n[ıi]z", re.IGNORECASE)),
    ("remaining_bir_olay_yasadin", re.compile(r"bir olay yaşad[ıi]n mı\?", re.IGNORECASE)),
    ("remaining_mutlu_eden_olay", re.compile(r"mutlu eden bir olay yaşadınız mı\?", re.IGNORECASE)),
    ("remaining_iyi_geliyor_musunuz", re.compile(r"iyi geliyor musunuz\?", re.IGNORECASE)),
    ("remaining_sevindiren_aniniz", re.compile(r"sevindiren bir anınızı", re.IGNORECASE)),
    ("remaining_evde_neden", re.compile(r"evde neden yapmaktan hoşlanırsınız", re.IGNORECASE)),
    ("remaining_prompt_peki", re.compile(r"\bPeki[^.?!]{0,80}\?", re.IGNORECASE)),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Conservatively repair Turkish ASR transcripts.")
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    parser.add_argument("--transcripts", default=DEFAULT_TRANSCRIPTS)
    parser.add_argument("--review-jsonl", default=str(DEFAULT_REVIEW_JSONL))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--report-md", default=str(DEFAULT_REPORT_MD))
    return parser.parse_args()


def load_review_rows(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            rows[str(payload["audio_filename"])] = payload
    return rows


def normalize_text(text: str) -> str:
    normalized = SPACE_BEFORE_PUNCT_RE.sub(r"\1", text)
    normalized = MULTISPACE_RE.sub(" ", normalized).strip()
    return normalized


def trim_incomplete_ellipsis_tail(text: str) -> tuple[str, bool]:
    if not text.endswith(ELLIPSIS_ENDINGS):
        return text, False
    last_boundary = max(text.rfind(". "), text.rfind("? "), text.rfind("! "))
    if last_boundary == -1:
        return text, False
    trimmed = text[: last_boundary + 1].strip()
    return trimmed, trimmed != text


def apply_replacements(text: str) -> tuple[str, list[str]]:
    actions: list[str] = []
    updated = text
    for label, pattern, replacement in TEXT_REPLACEMENTS:
        newer, count = pattern.subn(replacement, updated)
        if count:
            updated = newer
            actions.extend([label] * count)
    return updated, actions


def remove_prompt_fragments(text: str) -> tuple[str, list[str]]:
    actions: list[str] = []
    updated = text
    for label, pattern in PROMPT_REMOVALS:
        newer, count = pattern.subn("", updated)
        if count:
            updated = newer
            actions.extend([label] * count)
    return updated, actions


def remaining_prompt_reasons(text: str) -> list[str]:
    reasons: list[str] = []
    for label, pattern in REMAINING_PROMPT_PATTERNS:
        if pattern.search(text):
            reasons.append(label)
    return reasons


def repair_record(
    record: dict[str, Any],
    review_row: dict[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    basename = Path(str(record["audio_path"])).name
    original_text = str(record.get("transcript", "")).strip()
    repaired_text = original_text
    action_codes: list[str] = []
    manual_review_reasons: list[str] = []

    if basename in FAIL_FILENAMES:
        updated = dict(record)
        updated["transcript"] = ""
        updated["repair_status"] = "FAIL"
        updated["repair_actions"] = ["hard_fail_non_turkish_or_wrong_context"]
        updated["manual_review_recommended"] = False
        updated["manual_review_reason_codes"] = []
        updated["original_transcript"] = original_text
        audit = {
            "audio_filename": basename,
            "repair_status": "FAIL",
            "manual_review_recommended": False,
            "action_codes": ["hard_fail_non_turkish_or_wrong_context"],
            "manual_review_reason_codes": [],
            "original_transcript": original_text,
            "repaired_transcript": "",
        }
        return updated, audit

    trimmed_text, trimmed = trim_incomplete_ellipsis_tail(repaired_text)
    if trimmed:
        repaired_text = trimmed_text
        action_codes.append("trim_incomplete_ellipsis_tail")

    replaced_text, replacement_actions = apply_replacements(repaired_text)
    repaired_text = replaced_text
    action_codes.extend(replacement_actions)

    prompt_cleaned_text, prompt_actions = remove_prompt_fragments(repaired_text)
    repaired_text = prompt_cleaned_text
    action_codes.extend(prompt_actions)

    repaired_text = normalize_text(repaired_text)
    changed = repaired_text != original_text
    repair_status = "REPAIRED" if changed else "UNCHANGED"

    review_issue_codes = list((review_row or {}).get("issue_codes", []))
    if (review_row or {}).get("likely_continuation_fragment"):
        manual_review_reasons.append("likely_continuation_fragment")
    if repaired_text.endswith(ELLIPSIS_ENDINGS):
        manual_review_reasons.append("ends_with_ellipsis")
    if "repetition" in review_issue_codes:
        manual_review_reasons.append("repetition_flagged_in_review")
    if "too_short" in review_issue_codes and not changed:
        manual_review_reasons.append("short_low_information_clip")
    manual_review_reasons.extend(remaining_prompt_reasons(repaired_text))

    # Deduplicate while preserving order.
    dedup_action_codes: list[str] = []
    for code in action_codes:
        if code not in dedup_action_codes:
            dedup_action_codes.append(code)
    dedup_manual_review_reasons: list[str] = []
    for code in manual_review_reasons:
        if code not in dedup_manual_review_reasons:
            dedup_manual_review_reasons.append(code)

    updated = dict(record)
    updated["transcript"] = repaired_text
    updated["repair_status"] = repair_status
    updated["repair_actions"] = dedup_action_codes
    updated["manual_review_recommended"] = bool(dedup_manual_review_reasons)
    updated["manual_review_reason_codes"] = dedup_manual_review_reasons
    updated["original_transcript"] = original_text

    audit = {
        "audio_filename": basename,
        "repair_status": repair_status,
        "manual_review_recommended": bool(dedup_manual_review_reasons),
        "action_codes": dedup_action_codes,
        "manual_review_reason_codes": dedup_manual_review_reasons,
        "original_transcript": original_text,
        "repaired_transcript": repaired_text,
    }
    return updated, audit


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_audit_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fieldnames = [
        "audio_filename",
        "repair_status",
        "manual_review_recommended",
        "action_codes",
        "manual_review_reason_codes",
        "original_transcript",
        "repaired_transcript",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            rendered = dict(row)
            rendered["action_codes"] = "|".join(row["action_codes"])
            rendered["manual_review_reason_codes"] = "|".join(row["manual_review_reason_codes"])
            writer.writerow(rendered)


def build_markdown_report(
    report_path: Path,
    repaired_jsonl: Path,
    audit_csv: Path,
    summary_json: Path,
    audit_rows: list[dict[str, Any]],
) -> None:
    status_counts = Counter(row["repair_status"] for row in audit_rows)
    action_counts = Counter(code for row in audit_rows for code in row["action_codes"])
    manual_review_rows = [row for row in audit_rows if row["manual_review_recommended"]]
    manual_reason_counts = Counter(
        code for row in manual_review_rows for code in row["manual_review_reason_codes"]
    )
    lexical_rows = [
        row
        for row in audit_rows
        if row["repair_status"] != "FAIL"
        if any(code not in {"trim_incomplete_ellipsis_tail"} for code in row["action_codes"])
    ]
    fail_rows = [row for row in audit_rows if row["repair_status"] == "FAIL"]

    lines: list[str] = []
    lines.append("# Turkish Transcript Repair Report")
    lines.append("")
    lines.append(f"Date: {datetime.now(timezone.utc).date().isoformat()}")
    lines.append("")
    lines.append("## Outputs")
    lines.append("")
    lines.append(f"- Repaired JSONL: `{repaired_jsonl}`")
    lines.append(f"- Audit CSV: `{audit_csv}`")
    lines.append(f"- Summary JSON: `{summary_json}`")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Total transcripts processed: {len(audit_rows)}")
    lines.append(f"- `REPAIRED`: {status_counts.get('REPAIRED', 0)}")
    lines.append(f"- `UNCHANGED`: {status_counts.get('UNCHANGED', 0)}")
    lines.append(f"- `FAIL`: {status_counts.get('FAIL', 0)}")
    lines.append(f"- Manual review recommended: {len(manual_review_rows)}")
    lines.append("")
    lines.append("## Action Counts")
    lines.append("")
    for action, count in sorted(action_counts.items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"- `{action}`: {count}")
    lines.append("")
    lines.append("## FAIL Cases")
    lines.append("")
    if fail_rows:
        for row in fail_rows:
            lines.append(f"- `{row['audio_filename']}`")
            lines.append(f"  - original: `{row['original_transcript']}`")
    else:
        lines.append("- None")
    lines.append("")
    lines.append("## Lexical And Prompt Repairs")
    lines.append("")
    if lexical_rows:
        for row in lexical_rows:
            filtered_actions = [
                code for code in row["action_codes"] if code != "trim_incomplete_ellipsis_tail"
            ]
            if not filtered_actions:
                continue
            lines.append(f"- `{row['audio_filename']}` | `{', '.join(filtered_actions)}`")
            lines.append(f"  - before: `{row['original_transcript']}`")
            lines.append(f"  - after: `{row['repaired_transcript']}`")
    else:
        lines.append("- None")
    lines.append("")
    lines.append("## Ellipsis Tail Trims")
    lines.append("")
    trimmed_rows = [
        row for row in audit_rows if "trim_incomplete_ellipsis_tail" in row["action_codes"]
    ]
    lines.append(f"- Count: {len(trimmed_rows)}")
    if trimmed_rows:
        lines.append("- Files:")
        for row in trimmed_rows:
            lines.append(f"  - `{row['audio_filename']}`")
    lines.append("")
    lines.append("## Manual Review Queue")
    lines.append("")
    if manual_review_rows:
        lines.append("Files still carrying unresolved ambiguity after the conservative pass:")
        for row in manual_review_rows:
            reasons = ", ".join(row["manual_review_reason_codes"])
            lines.append(f"- `{row['audio_filename']}` | `{reasons}`")
    else:
        lines.append("- None")
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- Only exact or near-exact mechanical fixes were applied.")
    lines.append("- No semantic rewrites were attempted.")
    lines.append("- Hard failures were blanked in the repaired JSONL so downstream loaders can skip them.")
    lines.append("- Full row-level before/after detail is preserved in the audit CSV.")
    lines.append("")

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    root = Path(args.root).resolve()
    transcript_path = root / str(args.transcripts)
    review_path = Path(args.review_jsonl).resolve()
    output_dir = ensure_dir(Path(args.output_dir))
    report_md = Path(args.report_md).resolve()

    if not transcript_path.exists():
        raise FileNotFoundError(f"Transcript JSONL not found: {transcript_path}")
    if not review_path.exists():
        raise FileNotFoundError(f"Review JSONL not found: {review_path}")

    review_rows = load_review_rows(review_path)
    repaired_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []

    with transcript_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            basename = Path(str(record["audio_path"])).name
            repaired_record, audit = repair_record(record, review_rows.get(basename))
            repaired_rows.append(repaired_record)
            audit_rows.append(audit)

    repaired_jsonl = output_dir / "whisper_transcripts_repaired.jsonl"
    audit_csv = output_dir / "turkish_transcript_repair_audit.csv"
    summary_json = output_dir / "turkish_transcript_repair_summary.json"

    write_jsonl(repaired_rows, repaired_jsonl)
    write_audit_csv(audit_rows, audit_csv)

    status_counts = Counter(row["repair_status"] for row in audit_rows)
    action_counts = Counter(code for row in audit_rows for code in row["action_codes"])
    manual_reason_counts = Counter(
        code
        for row in audit_rows
        if row["manual_review_recommended"]
        for code in row["manual_review_reason_codes"]
    )
    summary_payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "total_transcripts": len(audit_rows),
        "repair_status_counts": dict(status_counts),
        "action_counts": dict(action_counts),
        "manual_review_count": sum(1 for row in audit_rows if row["manual_review_recommended"]),
        "manual_review_reason_counts": dict(manual_reason_counts),
        "fail_filenames": [
            row["audio_filename"] for row in audit_rows if row["repair_status"] == "FAIL"
        ],
    }
    summary_json.write_text(
        json.dumps(summary_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    build_markdown_report(
        report_path=report_md,
        repaired_jsonl=repaired_jsonl,
        audit_csv=audit_csv,
        summary_json=summary_json,
        audit_rows=audit_rows,
    )

    print(
        json.dumps(
            {
                "repaired_jsonl": str(repaired_jsonl),
                "audit_csv": str(audit_csv),
                "summary_json": str(summary_json),
                "report_md": str(report_md),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
