"""Freeze a local holdout, purging train neighbors linked to validation scenes.

Uses official train metadata, the local broad-group audit, and the team's
corrupted-train SIFT matches. Never reads test labels or edits original data.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


def read_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(train_path: Path, groups_path: Path, sift_path: Path, output: Path,
            fold: int = 2, minimum_inliers: int = 16) -> dict:
    train, groups, matches = (read_rows(p) for p in (train_path, groups_path, sift_path))
    if not (len(train) == len(groups) == len(matches)):
        raise ValueError("Input lengths differ; SIFT row indices require the official train order")
    if any(a["id"] != b["id"] or int(a["label"]) != int(b["label"])
           for a, b in zip(train, groups)):
        raise ValueError("Group metadata does not match official train row order")
    parents = list(range(len(train)))

    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    def union(i, j):
        a, b = find(i), find(j)
        parents[max(a, b)] = min(a, b)

    first = {}
    for i, row in enumerate(groups):
        group = row["broad_group"]
        if group in first:
            union(i, first[group])
        else:
            first[group] = i
    links = 0
    for i, row in enumerate(matches):
        if int(row["inlier"]) < minimum_inliers:
            continue
        j = int(row["idx"])
        if not 0 <= j < len(train) or i == j:
            raise ValueError("Invalid SIFT reference index")
        if int(row["label"]) != int(train[j]["label"]) or int(train[i]["label"]) != int(train[j]["label"]):
            raise ValueError("Strong SIFT label conflict requires review before freezing groups")
        union(i, j)
        links += 1
    validation = {i for i, row in enumerate(groups) if int(row["provisional_broad_fold"]) == fold}
    if not validation or len(validation) == len(train):
        raise ValueError("Selected fold must leave nonempty train and validation")
    validation_components = {find(i) for i in validation}
    manifest = []
    for i, row in enumerate(train):
        role = "validation" if i in validation else "purged_scene_neighbor" if find(i) in validation_components else "train"
        manifest.append({"row_index": i, "id": row["id"], "label": int(row["label"]), "role": role,
                         "strict_group": int(groups[i]["strict_group"]),
                         "broad_group": int(groups[i]["broad_group"]), "expanded_group": find(i)})
    retained_train = [r for r in manifest if r["role"] == "train"]
    heldout = [r for r in manifest if r["role"] == "validation"]
    purged = [r for r in manifest if r["role"] == "purged_scene_neighbor"]
    for key in ("strict_group", "broad_group", "expanded_group"):
        if {r[key] for r in retained_train} & {r[key] for r in heldout}:
            raise ValueError(f"Train/validation overlap remains in {key}")
    expected_labels = set(range(30))
    if {r["label"] for r in retained_train} != expected_labels or {r["label"] for r in heldout} != expected_labels:
        raise ValueError("Both partitions must cover all 30 classes")
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(manifest[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(manifest)
    content = buffer.getvalue().encode()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "split_manifest.csv"
    if manifest_path.exists() and manifest_path.read_bytes() != content:
        raise ValueError("Existing frozen manifest differs; use a new output directory")
    metadata = {
        "proposal_id": "LOCAL-PROP-001", "revision": 2,
        "protocol": "local_holdout_v4rot224_v1", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha256": {"train.csv": digest(train_path), "audit_groups": digest(groups_path), "sift_trainq": digest(sift_path)},
        "validation_source_broad_fold": fold, "minimum_sift_inliers": minimum_inliers,
        "accepted_train_sift_links": links, "expanded_groups": len({r["expanded_group"] for r in manifest}),
        "train_rows": len(retained_train), "validation_rows": len(heldout), "purged_rows": len(purged),
        "train_class_counts": dict(sorted(Counter(r["label"] for r in retained_train).items())),
        "validation_class_counts": dict(sorted(Counter(r["label"] for r in heldout).items())),
        "purged_ids": [r["id"] for r in purged], "detected_group_overlap": False,
        "manifest_sha256": hashlib.sha256(content).hexdigest(),
        "team_scene237_split_reproduced": False,
        "limitations": ["SIFT matches have implicit official row order and no coverage measurements.",
                        "Grouping remains provisional; undetected scene overlap is possible.",
                        "Scores use a local holdout and are not directly comparable with team OOF."],
    }
    metadata_path = output / "split_metadata.json"
    if manifest_path.exists() and metadata_path.exists():
        existing = json.loads(metadata_path.read_text())
        for key in metadata:
            if key != "created_at_utc" and existing.get(key) != json.loads(json.dumps(metadata[key])):
                raise ValueError(f"Frozen metadata differs in {key}; inspect before replacing")
        return existing
    manifest_path.write_bytes(content)
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=Path("data/raw/train.csv"))
    parser.add_argument("--groups", type=Path, default=Path("outputs/deep_audit/provisional_train_scene_groups.csv"))
    parser.add_argument("--sift", type=Path, default=Path("outputs/upstream/kaka/experiments/sift/sift_trainq.csv"))
    parser.add_argument("--output", type=Path, default=Path("outputs/LOCAL-PROP-001/plan"))
    args = parser.parse_args()
    result = prepare(args.train, args.groups, args.sift, args.output)
    print(json.dumps({key: result[key] for key in ("protocol", "train_rows", "validation_rows", "purged_rows", "manifest_sha256")}, indent=2))


if __name__ == "__main__":
    main()
