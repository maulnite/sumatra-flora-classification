"""Check the exact Kaggle CSV schema and organizer-provided ID order."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

EXPECTED_ROWS = 2010
EXPECTED_COLUMNS = ["id", "label"]


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]], list[str]]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as source:
            reader = csv.DictReader(source)
            columns = reader.fieldnames or []
            rows = list(reader)
    except (OSError, UnicodeError, csv.Error) as exc:
        return [], [], [f"Cannot read {path}: {exc}"]
    issues = []
    if columns != EXPECTED_COLUMNS:
        issues.append(f"Expected columns {EXPECTED_COLUMNS} in {path.name}, got {columns}")
    return columns, rows, issues


def validate(file_path: Path, template_path: Path) -> list[str]:
    _, rows, issues = read_csv(file_path)
    _, official, template_issues = read_csv(template_path)
    issues.extend(template_issues)
    if issues:
        return issues
    if len(rows) != EXPECTED_ROWS:
        issues.append(f"Expected {EXPECTED_ROWS} rows, found {len(rows)}")
    if len(official) != EXPECTED_ROWS:
        issues.append(f"Official template should have {EXPECTED_ROWS} rows, found {len(official)}")
    actual_ids = [r.get("id") for r in rows]
    official_ids = [r.get("id") for r in official]
    if any(not isinstance(x, str) or not x.strip() for x in actual_ids):
        issues.append("Missing or blank IDs")
    if len(set(actual_ids)) != len(actual_ids):
        issues.append("Duplicate IDs")
    if actual_ids != official_ids:
        issues.append("ID/order mismatch against sample_submission.csv")
    bad_labels = []
    for row_num, row in enumerate(rows, start=2):
        value = row.get("label")
        if value is None or not value.isdecimal() or not 0 <= int(value) <= 29:
            bad_labels.append((row_num, value))
    if bad_labels:
        issues.append(f"Invalid labels (must be integer 0..29): {bad_labels[:5]}")
    if any(None in row for row in rows):
        issues.append("Extra CSV fields found")
    return issues


def main() -> int:
    parser = argparse.ArgumentParser(description="Check a model-generated CSV against official template.")
    parser.add_argument("--file", type=Path, required=True)
    parser.add_argument("--template", type=Path, required=True)
    args = parser.parse_args()
    errors = validate(args.file, args.template)
    if errors:
        print("FAIL:")
        for error in errors:
            print(" -", error)
        return 1
    print("PASS: 2010 ordered IDs, two columns, integer labels 0..29")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
