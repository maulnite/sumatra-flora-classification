"""Local, model-free audit of scene overlap, train-label candidates and image quality.

OpenCV 4.8.1.78 can be isolated under outputs/audit_tools (see audit report).
All learned retrieval statistics use official training images only. Test labels
are never inferred or written. Original images and metadata remain untouched.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps
from scipy.fft import dctn
from scipy.stats import ks_2samp
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import StratifiedGroupKFold, StratifiedShuffleSplit
from threadpoolctl import threadpool_limits

SEED = 42
METRICS = (
    "brightness", "contrast", "saturation", "dark_fraction", "bright_fraction",
    "laplacian_var", "gradient_mean", "highpass_mad", "entropy", "native_block_excess",
)


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows, fields=None):
    if not rows and not fields:
        return
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def phashes(rgb):
    base = cv2.resize(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), (32, 32), interpolation=cv2.INTER_AREA)
    result = []
    for flip in (False, True):
        for turns in range(4):
            view = np.rot90(np.fliplr(base) if flip else base, turns)
            low = dctn(view.astype(np.float32), norm="ortho")[:8, :8].ravel()
            bits = low > np.median(low[1:])
            bits[0] = False
            result.append(np.packbits(bits))
    return np.stack(result)


def quality(rgb):
    small = cv2.resize(rgb, (256, 256), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY).astype(np.float32)
    hsv = cv2.cvtColor(small, cv2.COLOR_RGB2HSV)
    hist = np.bincount(gray.astype(np.uint8).ravel(), minlength=256).astype(float)
    hist /= hist.sum()
    native = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    boundary, internal = [], []
    for arr in (native, native.T):
        difference = np.abs(np.diff(arr, axis=1))
        boundary.append(float(difference[:, 7::8].mean()))
        mask = (np.arange(difference.shape[1]) + 1) % 8 != 0
        internal.append(float(difference[:, mask].mean()))
    residual = gray - cv2.GaussianBlur(gray, (0, 0), 1)
    return {
        "brightness": float(gray.mean()), "contrast": float(gray.std()),
        "saturation": float(hsv[:, :, 1].mean()) / 255,
        "dark_fraction": float((gray < 15).mean()),
        "bright_fraction": float((gray > 240).mean()),
        "laplacian_var": float(cv2.Laplacian(gray, cv2.CV_32F).var()),
        "gradient_mean": float(np.hypot(cv2.Sobel(gray, cv2.CV_32F, 1, 0), cv2.Sobel(gray, cv2.CV_32F, 0, 1)).mean()),
        "highpass_mad": float(np.median(np.abs(residual - np.median(residual)))) / 0.67449,
        "entropy": float(-(hist[hist > 0] * np.log2(hist[hist > 0])).sum()),
        "native_block_excess": float(np.mean(boundary) - np.mean(internal)),
    }


def extract_one(item):
    root, row, max_side, features = item
    path = root / row["path"]
    with Image.open(path) as image:
        orientation = int(image.getexif().get(274, 1))
        rgb = np.asarray(image.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    scale = min(1, max_side / max(gray.shape))
    gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    detector = cv2.SIFT_create(nfeatures=features, contrastThreshold=0.025, edgeThreshold=12)
    keypoints, descriptors = detector.detectAndCompute(gray, None)
    if descriptors is None:
        descriptors = np.zeros((0, 128), np.float32)
    points = np.array([k.pt for k in keypoints], dtype=np.float32).reshape(-1, 2)
    # RootSIFT is an analytic normalization, with no pretrained parameters.
    descriptors = np.sqrt(descriptors / np.maximum(descriptors.sum(axis=1, keepdims=True), 1e-9))
    hsv = cv2.cvtColor(cv2.resize(rgb, (128, 128), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2HSV)
    color = cv2.calcHist([hsv], [0, 1, 2], None, [12, 4, 4], [0, 180, 0, 256, 0, 256]).ravel()
    color = np.sqrt(color / max(float(color.sum()), 1))
    thumbnail = cv2.resize(rgb, (12, 12), interpolation=cv2.INTER_AREA).astype(np.float32).ravel() / 255
    return phashes(rgb), descriptors, points, color, thumbnail, {
        **row, "orientation": orientation, "keypoints": len(points),
        "sift_width": gray.shape[1], "sift_height": gray.shape[0], **quality(rgb),
    }


def extract(root, output, workers, max_side=448, features=240):
    rows = []
    for split, name in (("train", "train.csv"), ("test", "sample_submission.csv")):
        for row in read_csv(root / name):
            rows.append({"split": split, "id": row["id"], "path": split + "/" + row["id"],
                         "label": int(row["label"]) if split == "train" else ""})
    csv_hashes = {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                  for name in ("train.csv", "sample_submission.csv")}
    fingerprint = {"version": 1, "csv_hashes": csv_hashes, "max_side": max_side, "features": features, "seed": SEED}
    cache_path = output / "features.npz"
    config_path = output / "feature_config.json"
    if cache_path.exists() and config_path.exists() and json.loads(config_path.read_text()) == fingerprint:
        print("Reusing feature cache", flush=True)
        data = dict(np.load(cache_path, allow_pickle=False))
        return read_csv(output / "image_quality.csv"), data
    print(f"Extracting {len(rows)} images using {workers} workers", flush=True)
    all_hashes, all_desc, all_points, all_color, all_thumb, metadata = [], [], [], [], [], []
    offsets = [0]
    args = ((root, r, max_side, features) for r in rows)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for index, (hashes, desc, pts, color, thumb, row) in enumerate(pool.map(extract_one, args)):
            all_hashes.append(hashes); all_desc.append(desc); all_points.append(pts)
            all_color.append(color); all_thumb.append(thumb); metadata.append(row)
            offsets.append(offsets[-1] + len(desc))
            if (index + 1) % 500 == 0:
                print(f"  decoded/features {index + 1}/{len(rows)}", flush=True)
    data = {"hashes": np.stack(all_hashes), "descriptors": np.concatenate(all_desc),
            "points": np.concatenate(all_points), "colors": np.stack(all_color),
            "thumbnails": np.stack(all_thumb), "offsets": np.array(offsets, dtype=np.int64)}
    np.savez(cache_path, **data)
    write_json(config_path, fingerprint)
    write_csv(output / "image_quality.csv", metadata)
    return metadata, data


def hash_candidates(hashes, neighbors=6):
    count = len(hashes)
    lut = np.array([int(i).bit_count() for i in range(256)], np.uint8)
    candidates, near = set(), []
    min_dist = np.empty((count, count), np.uint8)
    print("Finding orientation-invariant perceptual-hash neighbors", flush=True)
    for start in range(0, count, 64):
        base = hashes[start:start + 64, 0]
        best = np.full((len(base), count), 64, np.uint8)
        for variant in range(8):
            distance = lut[np.bitwise_xor(base[:, None], hashes[None, :, variant])].sum(axis=2).astype(np.uint8)
            best = np.minimum(best, distance)
        for local, index in enumerate(range(start, start + len(base))):
            best[local, index] = 64
            top = np.argpartition(best[local], neighbors)[:neighbors]
            close = np.flatnonzero(best[local] <= 12)
            for other in np.unique(np.concatenate([top, close])):
                candidates.add(tuple(sorted((index, int(other)))))
        min_dist[start:start + len(base)] = best
    # Averaging DCT medians can cause tiny asymmetries; preserve the more similar result.
    min_dist = np.minimum(min_dist, min_dist.T)
    a, b = np.where(np.triu(min_dist <= 8, k=1))
    near = [(int(i), int(j), int(min_dist[i, j])) for i, j in zip(a, b)]
    return candidates, near, min_dist


def retrieve(data, ntrain, output):
    rng = np.random.default_rng(SEED)
    desc, offsets = data["descriptors"], data["offsets"]
    sample_size = min(40000, int(offsets[ntrain]))
    sample = desc[rng.choice(offsets[ntrain], sample_size, replace=False)]
    print("Fitting 128-word SIFT retrieval vocabulary on train only", flush=True)
    kmeans = MiniBatchKMeans(n_clusters=128, random_state=SEED, batch_size=2048,
                            n_init=3, max_iter=100, max_no_improvement=10, compute_labels=False)
    kmeans.fit(sample)
    words = kmeans.predict(desc)
    bow = np.zeros((len(offsets) - 1, 128), np.float32)
    for i in range(len(bow)):
        bow[i] = np.bincount(words[offsets[i]:offsets[i + 1]], minlength=128)
    transformer = TfidfTransformer(sublinear_tf=True)
    transformer.fit(bow[:ntrain])
    bow = transformer.transform(bow).toarray().astype(np.float32)
    np.savez(output / "retrieval_features.npz", vocabulary=kmeans.cluster_centers_, bow=bow, idf=transformer.idf_)
    return bow


def add_retrieval_candidates(candidates, feature, train_labels, ntrain, top=10):
    similarities = feature @ feature.T
    np.fill_diagonal(similarities, -1)
    for index, score in enumerate(similarities):
        others = np.argpartition(score, -top)[-top:]
        for other in others:
            candidates.add(tuple(sorted((index, int(other)))))
        if index >= ntrain:
            for other in np.argsort(score[:ntrain])[-top:]:
                candidates.add((int(other), index))
        if index < ntrain:
            # Also check likely neighbors within the same class, and across labels.
            for mask in (train_labels == train_labels[index], train_labels != train_labels[index]):
                eligible = np.flatnonzero(mask)
                eligible = eligible[eligible != index]
                if len(eligible):
                    selected = eligible[np.argsort(score[eligible])[-6:]]
                    for other in selected:
                        candidates.add(tuple(sorted((index, int(other)))))
    return similarities


def verify_pair(pair, data, rows, hash_dist, bow_sim, color_sim):
    i, j = pair
    offsets = data["offsets"]
    di = data["descriptors"][offsets[i]:offsets[i + 1]]
    dj = data["descriptors"][offsets[j]:offsets[j + 1]]
    if min(len(di), len(dj)) < 8:
        return None
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    forward = matcher.knnMatch(di, dj, k=2)
    good = [a for a, b in forward if a.distance < 0.75 * b.distance]
    if len(good) < 6:
        return None
    backward = matcher.knnMatch(dj, di, k=2)
    backward_pairs = {(a.trainIdx, a.queryIdx) for a, b in backward if a.distance < 0.75 * b.distance}
    matches = [m for m in good if (m.queryIdx, m.trainIdx) in backward_pairs]
    if len(matches) < 6:
        return None
    pi = data["points"][offsets[i]:offsets[i + 1]]
    pj = data["points"][offsets[j]:offsets[j + 1]]
    source = np.float32([pi[m.queryIdx] for m in matches])
    target = np.float32([pj[m.trainIdx] for m in matches])
    cv2.setRNGSeed((i * 100003 + j + SEED) % 2147483647)
    homography, mask = cv2.findHomography(source, target, cv2.RANSAC, 3.0, maxIters=1000, confidence=0.995)
    if homography is None or mask is None:
        return None
    inliers = mask.ravel().astype(bool)
    n = int(inliers.sum())
    if n < 6:
        return None
    coverage = []
    for points, row in ((source[inliers], rows[i]), (target[inliers], rows[j])):
        area = cv2.contourArea(cv2.convexHull(points))
        coverage.append(float(area / (int(row["sift_width"]) * int(row["sift_height"]))))
    projected = cv2.perspectiveTransform(source.reshape(-1, 1, 2), homography).reshape(-1, 2)
    error = float(np.median(np.linalg.norm(projected[inliers] - target[inliers], axis=1)))
    ratio = n / len(matches)
    # Conservative scene edges require numerous, spatially distributed geometric matches.
    strict = n >= 16 and ratio >= 0.5 and min(coverage) >= 0.025 and error <= 1.5
    broad = n >= 10 and ratio >= 0.35 and min(coverage) >= 0.01 and error <= 2
    if not broad and hash_dist[i, j] > 8:
        return None
    return {
        "i": i, "j": j, "path_a": rows[i]["path"], "path_b": rows[j]["path"],
        "split_a": rows[i]["split"], "split_b": rows[j]["split"],
        "phash_distance": int(hash_dist[i, j]), "ratio_matches": len(matches),
        "inliers": n, "inlier_ratio": ratio, "coverage_a": coverage[0], "coverage_b": coverage[1],
        "median_reprojection_error": error, "bow_cosine": float(bow_sim[i, j]),
        "color_cosine": float(color_sim[i, j]), "strict_scene": bool(strict), "broad_scene": bool(broad),
        "same_train_label": (int(rows[i]["label"]) == int(rows[j]["label"]))
        if rows[i]["split"] == rows[j]["split"] == "train" else "",
    }


def scene_groups(rows, pairs, criterion):
    ntrain = sum(r["split"] == "train" for r in rows)
    parents = list(range(ntrain))

    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    for pair in pairs:
        if pair[criterion] and pair["split_a"] == pair["split_b"] == "train":
            a, b = find(pair["i"]), find(pair["j"])
            parents[max(a, b)] = min(a, b)
    roots = [find(i) for i in range(ntrain)]
    unique = {root: index for index, root in enumerate(sorted(set(roots)))}
    groups = np.array([unique[r] for r in roots])
    sizes = Counter(groups.tolist())
    labels = np.array([int(r["label"]) for r in rows[:ntrain]])
    mixed = [g for g in sizes if len(set(labels[groups == g])) > 1]
    per_class = {str(label): int(len(set(groups[labels == label]))) for label in range(30)}
    summary = {
        "group_count": len(sizes), "non_singleton_groups": sum(v > 1 for v in sizes.values()),
        "images_in_non_singleton_groups": sum(v for v in sizes.values() if v > 1),
        "largest_group_sizes": sorted(sizes.values(), reverse=True)[:15],
        "mixed_label_groups": len(mixed), "groups_per_class": per_class,
    }
    estimates = []
    splitter = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=SEED)
    for train_idx, val_idx in splitter.split(np.zeros(ntrain), labels):
        overlap = np.isin(groups[val_idx], groups[train_idx])
        estimates.append(float(overlap.mean()))
    summary["random_80_20_validation_scene_overlap"] = {
        "mean": float(np.mean(estimates)), "min": min(estimates), "max": max(estimates),
        "n_splits": 10, "seed": SEED,
    }
    folds = np.full(ntrain, -1, int)
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    fold_info = []
    for fold, (tr, va) in enumerate(splitter.split(np.zeros(ntrain), labels, groups)):
        assert not set(groups[tr]) & set(groups[va])
        folds[va] = fold
        fold_info.append({"fold": fold, "train_images": len(tr), "val_images": len(va),
                          "classes_in_validation": len(set(labels[va])),
                          "val_class_counts": {str(c): int((labels[va] == c).sum()) for c in range(30)}})
    summary["provisional_group_5fold"] = fold_info
    return groups, folds, summary


def pair_sheet(root, output, rows, pairs, filename, max_pairs=12):
    # Only train/train examples may appear with labels in a review gallery.
    pairs = [p for p in pairs if p["split_a"] == p["split_b"] == "train"][:max_pairs]
    if not pairs:
        return
    sheet = Image.new("RGB", (720, 206 * len(pairs)), "white")
    draw = ImageDraw.Draw(sheet)
    for index, pair in enumerate(pairs):
        for column, key in enumerate(("i", "j")):
            row = rows[pair[key]]
            with Image.open(root / row["path"]) as source:
                thumb = ImageOps.contain(source.convert("RGB"), (350, 165))
            x, y = column * 360, index * 206
            sheet.paste(thumb, (x + (360 - thumb.width) // 2, y))
            draw.text((x + 3, y + 166), f"label {row['label']}  {Path(row['id']).name}", fill="black")
        draw.text((4, index * 206 + 184),
                  f"SIFT inliers={pair['inliers']} ratio={pair['inlier_ratio']:.2f} pHash={pair['phash_distance']}", fill="black")
    sheet.save(output / filename, quality=90)


def quality_controls(root, output, rows):
    rng = np.random.default_rng(SEED)
    train = [r for r in rows if r["split"] == "train"]
    test = [r for r in rows if r["split"] == "test"]
    sampled = []
    for label in range(30):
        eligible = [r for r in train if int(r["label"]) == label]
        sampled.extend(eligible[int(i)] for i in rng.choice(len(eligible), min(10, len(eligible)), replace=False))
    controls = []
    print(f"Measuring matched-JPEG controls on {len(sampled)} stratified train images", flush=True)
    for index, row in enumerate(sampled):
        encoding = test[int(rng.integers(len(test)))]
        with Image.open(root / encoding["path"]) as reference:
            tables = reference.quantization
        with Image.open(root / row["path"]) as image:
            buffer = io.BytesIO()
            image.convert("RGB").save(buffer, format="JPEG", qtables=tables, subsampling=2)
        buffer.seek(0)
        with Image.open(buffer) as image:
            rgb = np.asarray(image.convert("RGB"))
        controls.append({"split": "train_matched_jpeg", "source_train_id": row["id"],
                         "encoding_reference_test_id": encoding["id"], **quality(rgb)})
    write_csv(output / "jpeg_control_quality.csv", controls)
    summary = {}
    for metric in METRICS:
        arrays = {"train": np.array([float(r[metric]) for r in train]),
                  "test": np.array([float(r[metric]) for r in test]),
                  "train_matched_jpeg": np.array([float(r[metric]) for r in controls])}
        summary[metric] = {
            split: dict(zip(["p05", "median", "p95"], np.percentile(values, [5, 50, 95]).tolist()))
            for split, values in arrays.items()
        }
        summary[metric]["ks_train_test"] = float(ks_2samp(arrays["train"], arrays["test"]).statistic)
        summary[metric]["ks_jpeg_control_test"] = float(ks_2samp(arrays["train_matched_jpeg"], arrays["test"]).statistic)
    write_json(output / "quality_comparison.json", summary)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(2, 5, figsize=(19, 7))
    for ax, metric in zip(axes.ravel(), METRICS):
        for split, source, color in (("train", train, "#2563eb"), ("test", test, "#dc2626"),
                                     ("train JPEG control", controls, "#059669")):
            values = np.array([float(r[metric]) for r in source])
            if metric in {"laplacian_var", "gradient_mean", "highpass_mad"}:
                values = np.log1p(values)
            ax.hist(values, bins=40, density=True, histtype="step", linewidth=1.6, label=split, color=color)
        ax.set_title(metric + (" [log1p]" if metric in {"laplacian_var", "gradient_mean", "highpass_mad"} else ""))
        ax.grid(alpha=0.15)
    axes[0, 0].legend(fontsize=8)
    figure.suptitle("Official image quality: train / test / matched JPEG control (seed 42)")
    figure.tight_layout()
    figure.savefig(output / "quality_distributions.png", dpi=150)
    plt.close(figure)
    # Test-only gallery uses filenames and numeric quality metrics, without species labels.
    extremes = []
    for metric in ("brightness", "saturation", "laplacian_var", "entropy"):
        ordered = sorted(test, key=lambda r: float(r[metric]))
        extremes.extend((r, metric) for r in ordered[:3] + ordered[-3:])
    sheet = Image.new("RGB", (6 * 220, 4 * 190), "white")
    draw = ImageDraw.Draw(sheet)
    for index, (row, metric) in enumerate(extremes):
        with Image.open(root / row["path"]) as image:
            thumb = ImageOps.contain(image.convert("RGB"), (210, 150))
        x, y = index % 6 * 220, index // 6 * 190
        sheet.paste(thumb, (x + (220 - thumb.width) // 2, y))
        draw.text((x + 2, y + 151), row["id"], fill="black")
        draw.text((x + 2, y + 167), f"{metric}={float(row[metric]):.2f}", fill="black")
    sheet.save(output / "test_quality_extremes.jpg", quality=90)
    return summary


def train_outliers(rows, bow_sim, groups, output, root):
    ntrain = len(groups)
    labels = np.array([int(r["label"]) for r in rows[:ntrain]])
    result = []
    for i in range(ntrain):
        eligible = np.flatnonzero(groups != groups[i])
        nearest = eligible[np.argsort(bow_sim[i, eligible])[-10:][::-1]]
        votes = Counter(labels[nearest].tolist())
        other = [j for j in eligible if labels[j] != labels[i]]
        own = [j for j in eligible if labels[j] == labels[i]]
        if not own or not other:
            continue
        best_own = max(own, key=lambda j: bow_sim[i, j])
        best_other = max(other, key=lambda j: bow_sim[i, j])
        top_label, top_votes = votes.most_common(1)[0]
        result.append({
            "id": rows[i]["id"], "label": int(labels[i]), "scene_group": int(groups[i]),
            "own_label_votes_top10": votes.get(int(labels[i]), 0), "top_neighbor_label": top_label,
            "top_neighbor_label_votes": top_votes,
            "nearest_same_label_id": rows[best_own]["id"], "nearest_other_label_id": rows[best_other]["id"],
            "same_label_similarity": float(bow_sim[i, best_own]), "other_label_similarity": float(bow_sim[i, best_other]),
            "other_minus_own_similarity": float(bow_sim[i, best_other] - bow_sim[i, best_own]),
        })
    result.sort(key=lambda r: (r["top_neighbor_label"] != r["label"], r["top_neighbor_label_votes"],
                               r["other_minus_own_similarity"]), reverse=True)
    write_csv(output / "train_visual_outliers.csv", result)
    selected, class_counts = [], Counter()
    for row in result:
        if class_counts[row["label"]] < 2:
            selected.append(row); class_counts[row["label"]] += 1
        if len(selected) == 30:
            break
    sheet = Image.new("RGB", (6 * 220, 5 * 200), "white")
    draw = ImageDraw.Draw(sheet)
    for i, row in enumerate(selected):
        with Image.open(root / "train" / row["id"]) as image:
            thumb = ImageOps.contain(image.convert("RGB"), (210, 150))
        x, y = i % 6 * 220, i // 6 * 200
        sheet.paste(thumb, (x + (220 - thumb.width) // 2, y))
        draw.text((x + 2, y + 151), f"label {row['label']}  {Path(row['id']).name}", fill="black")
        draw.text((x + 2, y + 167), f"other vote {row['top_neighbor_label']} ({row['top_neighbor_label_votes']}/10)", fill="black")
        draw.text((x + 2, y + 183), f"similarity gap {row['other_minus_own_similarity']:.3f}", fill="black")
    sheet.save(output / "train_outlier_candidates.jpg", quality=90)
    return {"rows_ranked": len(result), "warning": "Retrieval disagreement is not a botanical label diagnosis; no labels changed."}


def analyze(root, output, rows, data, workers):
    ntrain = sum(r["split"] == "train" for r in rows)
    labels = np.array([int(r["label"]) for r in rows[:ntrain]])
    candidates, near, distances = hash_candidates(data["hashes"])
    write_csv(output / "phash_near_pairs.csv", [{"path_a": rows[i]["path"], "path_b": rows[j]["path"],
              "split_a": rows[i]["split"], "split_b": rows[j]["split"], "phash_distance": d}
              for i, j, d in near], ["path_a", "path_b", "split_a", "split_b", "phash_distance"])
    bow = retrieve(data, ntrain, output)
    bow_sim = add_retrieval_candidates(candidates, bow, labels, ntrain)
    color_sim = add_retrieval_candidates(candidates, data["colors"], labels, ntrain, top=6)
    # Normalized RGB thumbnails add candidates with similar composition.
    thumbs = data["thumbnails"] - data["thumbnails"].mean(axis=1, keepdims=True)
    thumbs /= np.maximum(np.linalg.norm(thumbs, axis=1, keepdims=True), 1e-9)
    add_retrieval_candidates(candidates, thumbs, labels, ntrain, top=4)
    print(f"Geometrically verifying {len(candidates)} candidate pairs using {workers} workers", flush=True)
    pairs = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        iterator = pool.map(lambda p: verify_pair(p, data, rows, distances, bow_sim, color_sim), sorted(candidates))
        for index, result in enumerate(iterator):
            if result:
                pairs.append(result)
            if (index + 1) % 10000 == 0:
                print(f"  verified {index + 1}/{len(candidates)}, retained {len(pairs)}", flush=True)
    pairs.sort(key=lambda r: (-r["inliers"], r["i"], r["j"]))
    pair_fields = ["i", "j", "path_a", "path_b", "split_a", "split_b", "phash_distance", "ratio_matches",
                   "inliers", "inlier_ratio", "coverage_a", "coverage_b", "median_reprojection_error", "bow_cosine",
                   "color_cosine", "strict_scene", "broad_scene", "same_train_label"]
    write_csv(output / "geometric_pairs.csv", pairs, pair_fields)
    strict_groups, folds, strict_summary = scene_groups(rows, pairs, "strict_scene")
    broad_groups, broad_folds, broad_summary = scene_groups(rows, pairs, "broad_scene")
    write_csv(output / "provisional_train_scene_groups.csv", [
        {"id": r["id"], "label": int(r["label"]), "strict_group": int(strict_groups[i]),
         "broad_group": int(broad_groups[i]), "provisional_fold": int(folds[i]),
         "provisional_broad_fold": int(broad_folds[i])}
        for i, r in enumerate(rows[:ntrain])])
    conflicts = [p for p in pairs if p["strict_scene"] and p["same_train_label"] is False]
    write_csv(output / "cross_label_scene_pairs.csv", conflicts, pair_fields)
    pair_sheet(root, output, rows, conflicts, "train_cross_label_scene_pairs.jpg")
    seen, scene_examples = set(), []
    for pair in pairs:
        if pair["strict_scene"] and pair["split_a"] == pair["split_b"] == "train":
            group = int(strict_groups[pair["i"]])
            if group not in seen:
                seen.add(group); scene_examples.append(pair)
    pair_sheet(root, output, rows, scene_examples, "train_scene_examples.jpg")
    outliers = train_outliers(rows, bow_sim, strict_groups, output, root)
    quality_summary = quality_controls(root, output, rows)
    cross_split = [p for p in pairs if p["strict_scene"] and p["split_a"] != p["split_b"]]
    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "seed": SEED,
        "opencv_version": cv2.__version__, "workers": workers,
        "images": len(rows), "train_images": ntrain, "test_images": len(rows) - ntrain,
        "phash_pairs_distance_at_most_8": len(near), "geometric_candidate_pairs": len(candidates),
        "retained_geometric_pairs": len(pairs), "strict_train_cross_label_pairs": len(conflicts),
        "strict_scene_groups": strict_summary, "broad_scene_groups": broad_summary,
        "strict_cross_split_pairs": len(cross_split),
        "train_images_in_cross_split_pairs": len({p["i"] for p in cross_split}),
        "test_images_in_cross_split_pairs": len({p["j"] for p in cross_split}),
        "orientation_counts": dict(Counter(str(r["orientation"]) for r in rows)),
        "label_review": outliers, "quality": quality_summary,
        "limitations": [
            "Candidate retrieval and geometric thresholds cannot prove absence of overlap.",
            "Connected components can merge through background features; groups require visual review.",
            "Train label outliers are heuristic candidates, not verified botanical errors.",
            "Quality proxies depend on scene content; JPEG controls do not establish other corruption causes.",
            "Test labels are not inferred or annotated. Original data are not modified.",
        ],
    }
    write_json(output / "deep_audit_summary.json", summary)
    print(json.dumps({k: summary[k] for k in ("phash_pairs_distance_at_most_8", "geometric_candidate_pairs",
          "retained_geometric_pairs", "strict_train_cross_label_pairs", "strict_cross_split_pairs",
          "test_images_in_cross_split_pairs")}, indent=2), flush=True)
    print("Strict group summary:", json.dumps(strict_summary), flush=True)


def expand_train(root, output, rows, data, workers):
    """Check every within-label training pair, beyond retrieval shortlist recall."""
    summary_path = output / "deep_audit_summary.json"
    if not summary_path.exists():
        raise ValueError("Run the main retrieval audit before --expand-train")
    summary = json.loads(summary_path.read_text())
    if "exhaustive_within_label_train_pairs" in summary:
        print("Within-label expansion already completed; keeping its recorded outputs.", flush=True)
        return
    original = read_csv(output / "geometric_pairs.csv")
    for filename in ("geometric_pairs.csv", "deep_audit_summary.json"):
        path = output / filename
        backup = output / (path.stem + "_retrieval" + path.suffix)
        if not backup.exists():
            backup.write_bytes(path.read_bytes())
    fields = list(original[0])
    pairs = []
    integers = {"i", "j", "phash_distance", "ratio_matches", "inliers"}
    floats = {"inlier_ratio", "coverage_a", "coverage_b", "median_reprojection_error", "bow_cosine", "color_cosine"}
    for row in original:
        pairs.append({key: int(value) if key in integers else float(value) if key in floats
                      else value == "True" if key in {"strict_scene", "broad_scene", "same_train_label"} and value != ""
                      else value for key, value in row.items()})
    ntrain = summary["train_images"]
    labels = np.array([int(r["label"]) for r in rows[:ntrain]])
    done = {(p["i"], p["j"]) for p in pairs}
    candidates, pair_space = [], 0
    lut = np.array([int(i).bit_count() for i in range(256)], np.uint8)
    distances = np.full((len(rows), len(rows)), 64, np.uint8)
    for label in range(30):
        selected = np.flatnonzero(labels == label)
        h = data["hashes"][selected]
        d = lut[np.bitwise_xor(h[:, None, None, 0, :], h[None, :, :, :])].sum(axis=3).min(axis=2)
        distances[np.ix_(selected, selected)] = np.minimum(d, d.T).astype(np.uint8)
        for a, i in enumerate(selected):
            for j in selected[a + 1:]:
                pair_space += 1
                if (int(i), int(j)) not in done:
                    candidates.append((int(i), int(j)))
    bow = np.load(output / "retrieval_features.npz")["bow"]
    bow_sim = bow @ bow.T
    color_sim = data["colors"] @ data["colors"].T
    print(f"Within-label train expansion: {pair_space} total pairs; {len(candidates)} verification calls", flush=True)
    retained = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        iterator = pool.map(lambda p: verify_pair(p, data, rows, distances, bow_sim, color_sim), candidates)
        for index, result in enumerate(iterator):
            if result:
                pairs.append(result); retained += 1
            if (index + 1) % 20000 == 0:
                print(f"  expanded {index + 1}/{len(candidates)}, extra retained {retained}", flush=True)
    pairs.sort(key=lambda r: (-r["inliers"], r["i"], r["j"]))
    write_csv(output / "geometric_pairs.csv", pairs, fields)
    strict, folds, strict_summary = scene_groups(rows, pairs, "strict_scene")
    broad, broad_folds, broad_summary = scene_groups(rows, pairs, "broad_scene")
    write_csv(output / "provisional_train_scene_groups.csv", [
        {"id": r["id"], "label": int(r["label"]), "strict_group": int(strict[i]),
         "broad_group": int(broad[i]), "provisional_fold": int(folds[i]),
         "provisional_broad_fold": int(broad_folds[i])}
        for i, r in enumerate(rows[:ntrain])])
    seen, examples = set(), []
    for pair in pairs:
        if pair["strict_scene"] and pair["split_a"] == pair["split_b"] == "train":
            group = int(strict[pair["i"]])
            if group not in seen:
                seen.add(group); examples.append(pair)
    pair_sheet(root, output, rows, examples, "train_scene_examples.jpg")
    summary.update({"created_at_utc": datetime.now(timezone.utc).isoformat(),
                    "exhaustive_within_label_train_pairs": pair_space,
                    "expanded_verification_calls": len(candidates), "extra_retained_pairs": retained,
                    "retained_geometric_pairs": len(pairs),
                    "strict_scene_groups": strict_summary, "broad_scene_groups": broad_summary,
                    "label_review": train_outliers(rows, bow_sim, strict, output, root)})
    write_json(summary_path, summary)
    print("Expanded strict scene groups:", json.dumps(strict_summary), flush=True)


def scan_one_artifacts(item):
    root, row = item
    with Image.open(root / row["path"]) as image:
        rgb = np.asarray(image.convert("RGB"))
    height, width = rgb.shape[:2]
    channel_range = rgb.max(axis=2).astype(np.int16) - rgb.min(axis=2).astype(np.int16)
    black = (rgb.max(axis=2) <= 6).astype(np.uint8)
    _, _, stats, _ = cv2.connectedComponentsWithStats(black, connectivity=8)
    interior, borders = [], 0
    for x, y, w, h, area in stats[1:]:
        if min(w, h) < 12 or area < height * width * 0.00015:
            continue
        if x == 0 or y == 0 or x + w >= width or y + h >= height:
            borders += 1
        elif area / (w * h) >= 0.85 and max(w / h, h / w) <= 8:
            interior.append([int(x), int(y), int(w), int(h)])
    return {
        "split": row["split"], "id": row["id"], "path": row["path"],
        "near_grayscale": bool((channel_range <= 2).mean() >= 0.995),
        "grayscale_pixel_fraction": float((channel_range <= 2).mean()),
        "interior_black_rectangles": len(interior), "large_black_border_components": borders,
        "black_rectangle_area_fraction": float(sum(w * h for _, _, w, h in interior) / (height * width)),
        "rectangles_xywh": json.dumps(interior),
    }


def artifact_scan(root, output, rows, workers):
    print("Scanning grayscale images and black-rectangle candidates", flush=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        scanned = list(pool.map(scan_one_artifacts, ((root, row) for row in rows)))
    write_csv(output / "digital_artifact_candidates.csv", scanned)
    summary = {}
    for split in ("train", "test"):
        subset = [r for r in scanned if r["split"] == split]
        summary[split] = {
            "images": len(subset), "near_grayscale_images": sum(r["near_grayscale"] for r in subset),
            "images_with_interior_black_rectangle_candidates": sum(r["interior_black_rectangles"] > 0 for r in subset),
            "interior_black_rectangle_candidates": sum(r["interior_black_rectangles"] for r in subset),
            "images_with_large_black_border_components": sum(r["large_black_border_components"] > 0 for r in subset),
        }
    write_json(output / "digital_artifact_summary.json", summary)
    summary_path = output / "deep_audit_summary.json"
    if summary_path.exists():
        combined = json.loads(summary_path.read_text())
        combined["digital_artifacts"] = summary
        write_json(summary_path, combined)
    rng = np.random.default_rng(SEED)
    positives = [r for r in scanned if r["split"] == "test" and r["interior_black_rectangles"]]
    selected = [positives[i] for i in rng.choice(len(positives), min(24, len(positives)), replace=False)]
    if selected:
        sheet = Image.new("RGB", (6 * 220, 4 * 190), "white")
        draw = ImageDraw.Draw(sheet)
        for i, row in enumerate(selected):
            with Image.open(root / row["path"]) as image:
                image = image.convert("RGB")
                marks = ImageDraw.Draw(image)
                for x, y, w, h in json.loads(row["rectangles_xywh"]):
                    marks.rectangle((x, y, x + w, y + h), outline="#00ff66", width=3)
                thumb = ImageOps.contain(image, (210, 150))
            x, y = i % 6 * 220, i // 6 * 190
            sheet.paste(thumb, (x + (220 - thumb.width) // 2, y))
            draw.text((x + 2, y + 151), row["id"], fill="black")
            draw.text((x + 2, y + 167), f"black rectangles: {row['interior_black_rectangles']}", fill="black")
        sheet.save(output / "test_black_rectangle_candidates.jpg", quality=90)
    # Train-only contexts for reviewing retrieval disagreements without relabeling.
    candidates = read_csv(output / "train_visual_outliers.csv")
    selected, used = [], set()
    for row in candidates:
        if row["label"] not in used:
            used.add(row["label"]); selected.append(row)
        if len(selected) == 12:
            break
    sheet = Image.new("RGB", (3 * 300, len(selected) * 210), "white")
    draw = ImageDraw.Draw(sheet)
    titles = ("candidate", "same-label nearest", "other-label nearest")
    for i, row in enumerate(selected):
        for column, key in enumerate(("id", "nearest_same_label_id", "nearest_other_label_id")):
            with Image.open(root / "train" / row[key]) as image:
                thumb = ImageOps.contain(image.convert("RGB"), (290, 170))
            x, y = column * 300, i * 210
            sheet.paste(thumb, (x + (300 - thumb.width) // 2, y))
            label = row[key].split("_")[0]
            draw.text((x + 3, y + 171), f"{titles[column]} | label {label}", fill="black")
            draw.text((x + 3, y + 188), Path(row[key]).name, fill="black")
    sheet.save(output / "train_outlier_contexts.jpg", quality=90)
    # A complete overview of the visually heterogeneous class seen in scene pairs.
    for label in (18, 17):
        members = [r for r in rows if r["split"] == "train" and int(r["label"]) == label]
        for page, start in enumerate(range(0, len(members), 30), 1):
            subset = members[start:start + 30]
            sheet = Image.new("RGB", (6 * 200, 5 * 175), "white")
            draw = ImageDraw.Draw(sheet)
            for i, row in enumerate(subset):
                with Image.open(root / row["path"]) as image:
                    thumb = ImageOps.contain(image.convert("RGB"), (190, 150))
                x, y = i % 6 * 200, i // 6 * 175
                sheet.paste(thumb, (x + (200 - thumb.width) // 2, y))
                draw.text((x + 3, y + 151), Path(row["id"]).name, fill="black")
            sheet.save(output / f"train_class_{label}_page_{page}.jpg", quality=90)
    print(json.dumps(summary, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/raw"))
    parser.add_argument("--output", type=Path, default=Path("outputs/deep_audit"))
    parser.add_argument("--workers", type=int, default=4)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--extract-only", action="store_true")
    mode.add_argument("--expand-train", action="store_true")
    mode.add_argument("--scan-artifacts", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cv2.setNumThreads(1)
    cv2.setRNGSeed(SEED)
    with threadpool_limits(limits=args.workers):
        rows, data = extract(args.root, args.output, args.workers)
        if args.extract_only:
            return
        if args.expand_train:
            expand_train(args.root, args.output, rows, data, args.workers)
            return
        if args.scan_artifacts:
            artifact_scan(args.root, args.output, rows, args.workers)
            return
        # Analysis is kept in a separate function to reuse cached features.
        analyze(args.root, args.output, rows, data, args.workers)


if __name__ == "__main__":
    main()
