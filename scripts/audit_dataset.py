"""Read official images and save a reproducible local dataset audit.

No training, prediction, test annotation, or remote writes are performed.
Outputs belong under the git-ignored outputs/ directory.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def audit(root: Path, output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    train = read_csv(root / "train.csv")
    sample = read_csv(root / "sample_submission.csv")
    mapping = read_csv(root / "class_mapping.csv")
    folders = {int(row["label"]): row["folder_name"] for row in mapping}
    counts = Counter(int(row["label"]) for row in train)
    expected = {"train/" + row["id"] for row in train}
    expected.update("test/" + row["id"] for row in sample)
    actual = {
        p.relative_to(root).as_posix()
        for split in ("train", "test")
        for p in (root / split).rglob("*")
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    }
    checks = {
        "train_ids_unique": len({r["id"] for r in train}) == len(train),
        "test_ids_unique": len({r["id"] for r in sample}) == len(sample),
        "labels_exactly_0_to_29": set(counts) == set(range(30)),
        "mapping_labels_unique": len(folders) == len(mapping) == 30,
        "train_folder_matches_label": all(
            Path(r["id"]).parts[0] == folders[int(r["label"])] for r in train
        ),
        "template_ids_in_official_order": [r["id"] for r in sample]
        == [f"test_{i:04d}.jpg" for i in range(1, 2011)],
        "all_metadata_images_exist": expected <= actual,
        "no_unlisted_images": actual <= expected,
    }
    groups = {"file_sha256": defaultdict(list), "pixel_sha256": defaultdict(list)}
    inventory, errors = [], []
    records = [("train", r) for r in train] + [("test", r) for r in sample]
    for split, row in records:
        relative = split + "/" + row["id"]
        path = root / relative
        try:
            with Image.open(path) as image:
                image.load()  # Decode every image, not just the header.
                width, height = image.size
                mode, fmt = image.mode, image.format
                luminance_table = getattr(image, "quantization", {}).get(0, [])
                rgb = image.convert("RGB")
                pixel_hash = hashlib.sha256(
                    f"{width}x{height}:RGB:".encode() + rgb.tobytes()
                ).hexdigest()
            file_hash = hashlib.sha256(path.read_bytes()).hexdigest()
            record = {
                "split": split, "id": row["id"],
                "label": int(row["label"]) if split == "train" else "",
                "width": width, "height": height, "mode": mode,
                "format": fmt, "bytes": path.stat().st_size,
                "bytes_per_pixel": path.stat().st_size / (width * height),
                "luminance_quantization_mean": (
                    float(np.mean(luminance_table)) if luminance_table else None
                ),
                "file_sha256": file_hash, "pixel_sha256": pixel_hash,
            }
            inventory.append(record)
            for key, values in groups.items():
                values[record[key]].append({"path": relative, "label": record["label"]})
        except Exception as exc:
            errors.append({"path": relative, "error": str(exc)})
    duplicates = {}
    for key, values in groups.items():
        duplicate_groups = [v for v in values.values() if len(v) > 1]
        duplicates[key] = {
            "groups": duplicate_groups,
            "group_count": len(duplicate_groups),
            "extra_copies": sum(len(g) - 1 for g in duplicate_groups),
            "cross_split_groups": sum(
                len({r["path"].split("/")[0] for r in g}) > 1 for g in duplicate_groups
            ),
            "conflicting_train_label_groups": sum(
                len({r["label"] for r in g if r["label"] != ""}) > 1
                for g in duplicate_groups
            ),
        }
    image_stats = {}
    for split in ("train", "test"):
        rows = [r for r in inventory if r["split"] == split]
        stats = {
            "decoded_images": len(rows),
            "modes": dict(Counter(r["mode"] for r in rows)),
            "formats": dict(Counter(r["format"] for r in rows)),
            "most_common_sizes": Counter(f"{r['width']}x{r['height']}" for r in rows).most_common(10),
        }
        for key in ("width", "height", "bytes", "bytes_per_pixel", "luminance_quantization_mean"):
            values = np.array([r[key] for r in rows if r[key] is not None])
            stats[key] = dict(zip(
                ["min", "p25", "median", "p75", "max"],
                np.percentile(values, [0, 25, 50, 75, 100]).tolist(),
            )) if len(values) else {}
        image_stats[split] = stats
    checks["all_images_decode"] = not errors
    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_root": str(root.resolve()),
        "python": sys.executable, "python_version": platform.python_version(),
        "train_rows": len(train), "test_rows": len(sample), "classes": len(mapping),
        "per_class": [{**r, "count": counts[int(r["label"])]} for r in mapping],
        "class_imbalance_max_over_min": max(counts.values()) / min(counts.values()),
        "checks": checks, "images": image_stats,
        "duplicates": duplicates, "decode_errors": errors,
        "missing_images": sorted(expected - actual),
        "unlisted_images": sorted(actual - expected),
        "csv_sha256": {
            name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in ("train.csv", "class_mapping.csv", "sample_submission.csv")
        },
    }
    with (output / "image_inventory.csv").open("w", newline="", encoding="utf-8") as stream:
        if inventory:
            writer = csv.DictWriter(stream, fieldnames=list(inventory[0]))
            writer.writeheader()
            writer.writerows(inventory)
    (output / "dataset_audit.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    # Training images only; test images never receive human annotations.
    sheet = Image.new("RGB", (6 * 240, 5 * 220), "white")
    draw = ImageDraw.Draw(sheet)
    for index, row in enumerate(sorted(mapping, key=lambda r: int(r["label"]))):
        label = int(row["label"])
        example = next(r for r in train if int(r["label"]) == label)
        with Image.open(root / "train" / example["id"]) as image:
            thumb = ImageOps.contain(image.convert("RGB"), (230, 180))
        x, y = (index % 6) * 240, (index // 6) * 220
        sheet.paste(thumb, (x + (240 - thumb.width) // 2, y))
        draw.text((x + 4, y + 183), f"{label}: {row['species'][:29]}", fill="black")
        draw.text((x + 4, y + 199), f"train n={counts[label]}", fill="black")
    sheet.save(output / "train_contact_sheet.jpg", quality=90)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/raw"))
    parser.add_argument("--output", type=Path, default=Path("outputs/discovery"))
    args = parser.parse_args()
    summary = audit(args.root, args.output)
    print(json.dumps({
        key: summary[key] for key in
        ("train_rows", "test_rows", "classes", "checks", "images")
    }, indent=2))
    print("Duplicate groups:", {k: v["group_count"] for k, v in summary["duplicates"].items()})
    return 0 if all(summary["checks"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
