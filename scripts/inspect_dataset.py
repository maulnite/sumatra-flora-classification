"""Describe organizer files on the local filesystem, without uploading them."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def inspect(root: Path) -> dict:
    train, test = root / "train", root / "test"
    class_dirs = sorted((x for x in train.iterdir() if x.is_dir())) if train.is_dir() else []
    per_class = {
        cls.name: sum(1 for p in cls.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
        for cls in class_dirs
    }
    test_names = sorted(
        p.name for p in test.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    ) if test.is_dir() else []
    csv_details = {}
    for name in ("train.csv", "class_mapping.csv", "sample_submission.csv"):
        path = root / name
        if not path.is_file():
            csv_details[name] = {"missing": True}
            continue
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.reader(stream)
            csv_details[name] = {"columns": next(reader, []), "rows": sum(1 for _ in reader)}
    return {
        "train_class_dirs": len(class_dirs),
        "train_images": sum(per_class.values()),
        "per_class": per_class,
        "test_images": len(test_names),
        "test_names": test_names,
        "csv": csv_details,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect official local competition data.")
    parser.add_argument("--root", type=Path, default=Path("data/raw"))
    args = parser.parse_args()
    if not args.root.exists():
        parser.error(f"Folder missing: {args.root}. Download official files first.")
    summary = inspect(args.root)
    print(f"Training classes: {summary['train_class_dirs']} (expected 30)")
    print(f"Training images: {summary['train_images']}")
    print(f"Test images: {summary['test_images']} (expected 2010)")
    for name, count in summary["per_class"].items():
        print(f"  {name}: {count}")
    for name, detail in summary["csv"].items():
        print(f"  {name}: {detail}")
    if summary["train_class_dirs"] != 30 or summary["test_images"] != 2010:
        print("WARNING: counts do not match organizer description; verify extraction.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
