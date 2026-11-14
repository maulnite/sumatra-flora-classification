"""Frozen two-fold embedding/retrieval/graph pilot; no backbone training or upload.

Primary queries read an already-built train+test graph independently. Secondary
queries enter a joint transductive graph without labels. Only source labels are
ever passed into graph construction and diffusion.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import importlib.metadata
import json
import os
from pathlib import Path
import time

import cv2
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import f1_score
import torch
import yaml

try:
    from . import reproduce_checkpoint_inference as replay
except ImportError:
    import reproduce_checkpoint_inference as replay

ROOT = Path(__file__).resolve().parents[1]


def unit(x):
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def probabilities(x):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or not np.isfinite(x).all() or (x < 0).any() or not np.allclose(x.sum(1), 1, atol=1e-5):
        raise ValueError("Invalid probability matrix")
    return x


def nearest(query, reference, k, same_set=False, chunk=256):
    """Exact cosine neighbors; source/query embeddings are unit-normalized."""
    k = min(k, len(reference) - int(same_set))
    if k < 1:
        raise ValueError("Not enough reference nodes")
    all_indices, all_similarities = [], []
    for start in range(0, len(query), chunk):
        similarity = query[start:start + chunk] @ reference.T
        if same_set:
            similarity[np.arange(len(similarity)), np.arange(start, start + len(similarity))] = -np.inf
        indices = np.argpartition(-similarity, k - 1, axis=1)[:, :k]
        values = np.take_along_axis(similarity, indices, axis=1)
        order = np.argsort(-values, axis=1, kind="stable")
        all_indices.append(np.take_along_axis(indices, order, axis=1))
        all_similarities.append(np.take_along_axis(values, order, axis=1))
    return np.concatenate(all_indices), np.concatenate(all_similarities)


def distinct_neighbors(query, gallery, source_ids, k):
    candidates, values = nearest(query, gallery, min(len(gallery), 2 * k))
    selected, similarities = [], []
    for row, scores in zip(candidates, values):
        seen, chosen, chosen_scores = set(), [], []
        for index, score in zip(row, scores):
            identity = int(source_ids[index])
            if identity not in seen:
                chosen.append(index); chosen_scores.append(score); seen.add(identity)
            if len(chosen) == k:
                break
        if len(chosen) != k:
            raise ValueError("Not enough distinct gallery images")
        selected.append(chosen); similarities.append(chosen_scores)
    return np.asarray(selected), np.asarray(similarities)


def weighted_labels(indices, similarities, values, temperature, valid=None):
    weights = np.exp((similarities - similarities.max(1, keepdims=True)) / temperature)
    if valid is not None:
        weights *= valid
    totals = weights.sum(1, keepdims=True)
    weights /= np.maximum(totals, 1e-30)
    return np.einsum("nk,nkc->nc", weights, values[indices]), totals[:, 0] > 0


def make_graph(embedding, cfg):
    indices, similarities = nearest(embedding, embedding, cfg["neighbors"], same_set=True)
    weights = np.exp((similarities - 1) / cfg["edge_temperature"])
    rows = np.repeat(np.arange(len(embedding)), indices.shape[1])
    graph = sparse.csr_matrix((weights.ravel(), (rows, indices.ravel())), shape=(len(embedding), len(embedding)))
    if cfg["mutual_knn"]:
        graph = graph.minimum(graph.T).tocsr()
    totals = np.asarray(graph.sum(1)).ravel()
    graph = sparse.diags(1 / np.maximum(totals, 1e-30)) @ graph
    return graph.tocsr(), similarities[:, -1], totals == 0


def diffuse(graph, unary, source_labels, cfg):
    unary = probabilities(unary).copy()
    source_count = len(source_labels)
    source = np.eye(unary.shape[1])[source_labels]
    unary[:source_count] = source
    result = unary.copy()
    isolated = np.asarray(graph.sum(1)).ravel() == 0
    change = None
    for iteration in range(cfg["iterations"]):
        updated = (1 - cfg["diffusion_alpha"]) * unary + cfg["diffusion_alpha"] * graph.dot(result)
        updated[isolated] = unary[isolated]
        updated[:source_count] = source
        change = float(np.abs(updated - result).max())
        result = updated
        if change <= cfg["tolerance"]:
            break
    probabilities(result)
    return result, dict(iterations=iteration + 1, maximum_last_change=change, isolated_nodes=int(isolated.sum()))


def independent_readout(query, embedding, graph_values, unary, kth_similarity, cfg):
    indices, similarities = nearest(query, embedding, cfg["neighbors"])
    valid = similarities >= cfg["independent_query_minimum_similarity"]
    if cfg["independent_query_reciprocal_neighbors"]:
        valid &= similarities >= kth_similarity[indices]
    neighbor_prob, connected = weighted_labels(indices, similarities, graph_values, cfg["edge_temperature"], valid)
    weight = cfg["final_neural_graph_weight"]
    result = (1 - weight) * unary + weight * neighbor_prob
    result[~connected] = unary[~connected]
    probabilities(result)
    return result, dict(connected_queries=int(connected.sum()), disconnected_queries=int((~connected).sum()))


def save_npz(path, **arrays):
    temporary = path.with_suffix(".tmp.npz")
    np.savez(temporary, **arrays)
    temporary.replace(path)


def clean_cache(size, cfg, out):
    destination = out / f"clean_{size}.npy"
    if destination.exists():
        return np.load(destination, mmap_mode="r")
    train = pd.read_csv(ROOT / cfg["dataset_root"] / "train.csv")
    arr = np.lib.format.open_memmap(destination.with_suffix(".tmp.npy"), mode="w+", dtype=np.uint8,
                                   shape=(len(train), size, size, 3))
    cv2.setNumThreads(1)
    def read(identity):
        image = cv2.imread(str(ROOT / cfg["dataset_root"] / "train" / identity))
        if image is None:
            raise ValueError(f"Unreadable official image {identity}")
        return cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), (size, size), interpolation=cv2.INTER_AREA)
    with ThreadPoolExecutor(cfg["features"]["preprocessing_workers"]) as pool:
        for index, image in enumerate(pool.map(read, train.id)):
            arr[index] = image
    arr.flush(); del arr
    destination.with_suffix(".tmp.npy").replace(destination)
    replay.log(f"Clean RGB cache {size} ready")
    return np.load(destination, mmap_mode="r")


def extract(model, images, indices, reference, device, batch_size, retry_seconds=20, progress_path=None):
    features, predicted = [], []
    mean = torch.tensor(reference.dq.MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(reference.dq.STD, device=device).view(1, 3, 1, 1)
    with torch.inference_mode():
        start = 0
        next_progress = 512
        while start < len(indices):
            batch = np.array(images[indices[start:start + batch_size]], copy=True)
            x = z = p = z_sum = p_sum = None
            try:
                x = torch.from_numpy(batch).to(device).permute(0, 3, 1, 2)
                x = ((x.float() - mean) / std).contiguous(memory_format=torch.channels_last)
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    for flip in (False, True):
                        z = model.forward_head(model.forward_features(x.flip(3) if flip else x), pre_logits=True)
                        p = model.get_classifier()(z).float().softmax(1)
                        z_sum = z.float() if z_sum is None else z_sum + z.float()
                        p_sum = p if p_sum is None else p_sum + p
            except torch.cuda.OutOfMemoryError:
                x = z = p = z_sum = p_sum = None
                torch.cuda.empty_cache()
                batch_size = max(1, batch_size // 2)
                replay.log(f"GPU allocation unavailable; retry batch={batch_size} in {retry_seconds}s, row={start}")
                if progress_path:
                    state = json.loads(progress_path.read_text())
                    state.update(waiting_for_gpu_memory=True, processed=start, total=len(indices), batch_size=batch_size, at=replay.now())
                    replay.write_json(progress_path, state)
                time.sleep(retry_seconds)
                continue
            features.append((z_sum / 2).cpu().numpy()); predicted.append((p_sum / 2).cpu().numpy())
            start += len(batch)
            if start >= next_progress or start == len(indices):
                replay.log(f"Feature rows {start}/{len(indices)}, batch={batch_size}, device={device}")
                if progress_path:
                    state = json.loads(progress_path.read_text())
                    state.update(waiting_for_gpu_memory=False, processed=start, total=len(indices), batch_size=batch_size, at=replay.now())
                    replay.write_json(progress_path, state)
                next_progress += 512
    return unit(np.concatenate(features)), np.concatenate(predicted)


def prepare_features(cfg, out, train_indices, validation_indices, fold, member, reference, evidence):
    directory = out / "features"; directory.mkdir(exist_ok=True)
    path = directory / f"{member['run']}_f{fold}.npz"
    if path.exists():
        replay.log(f"Reuse frozen feature file {path.name}")
        return dict(np.load(path))
    size = member["inference_size"]
    clean = clean_cache(size, cfg, out)
    team = Path(cfg["team_root"])
    corrupt = np.load(team / f"cache/corr4r_{size}.npy", mmap_mode="r")
    test = np.load(team / f"cache/test_{size}.npy", mmap_mode="r")
    original = team / "runs" / f"{member['run']}_f{fold}"
    model = reference.make_model(json.loads((original / "cfg.json").read_text()), 30)
    model.load_state_dict(torch.load(original / "model.pt", map_location="cpu", weights_only=True), strict=True)
    device = torch.device("cpu")
    if torch.cuda.is_available():
        free, _ = torch.cuda.mem_get_info()
        if free >= cfg["features"]["minimum_free_gpu_mib"] * 1024**2:
            device = torch.device("cuda")
            torch.cuda.set_per_process_memory_fraction(cfg["features"]["gpu_memory_fraction"])
    if device.type == "cpu" and not cfg["features"]["cpu_fallback"]:
        raise RuntimeError("GPU free memory below frozen minimum")
    while True:
        try:
            model = model.to(device).to(memory_format=torch.channels_last).eval()
            break
        except torch.cuda.OutOfMemoryError:
            model = model.cpu()
            torch.cuda.empty_cache()
            replay.log("Waiting for GPU model allocation; retrying without changing other processes")
            time.sleep(cfg["features"]["oom_retry_seconds"])
    arrays = {}
    for name, images, indices in (("clean", clean, train_indices), ("corrupt", corrupt, train_indices),
                                  ("query", corrupt, validation_indices), ("test", test, np.arange(len(test)))):
        replay.write_json(out / "status.json", dict(status="feature_extraction", fold=fold, member=member["run"], view=name,
                                                    device=str(device), at=replay.now(), pid=os.getpid()))
        started = time.monotonic()
        arrays[name], p = extract(model, images, indices, reference, device, cfg["features"]["batch_size"],
                                  cfg["features"]["oom_retry_seconds"], out / "status.json")
        if name in ("query", "test"):
            kind = "validation" if name == "query" else "test"
            expected = np.load(ROOT / cfg["reference_root"] / "predictions" / f"{member['run']}_f{fold}" / f"{kind}.npy")
            check = replay.compare_probabilities(p, expected, 1e-5, 1e-4)
            evidence[f"{member['run']}_f{fold}_{name}"] = check
            # Small-batch BF16 may differ numerically; unaries always use exact reference arrays.
            if check["predicted_labels_different"] > max(2, int(.005 * len(p))) or check["max_absolute_difference"] > .02:
                raise ValueError(f"Unexpected feature-head disagreement: {check}")
        replay.log(f"{member['run']} fold {fold} {name}: {len(indices)} embeddings on {device}, {time.monotonic()-started:.1f}s")
    save_npz(path, **arrays)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return arrays


def member_features(arrays):
    gallery = np.concatenate([arrays["clean"], arrays["corrupt"]])
    center = gallery.mean(0, keepdims=True)
    return {name: unit(value - center) for name, value in arrays.items()}


def solve_fold(cfg, combined, source_labels, base_query, base_test):
    gc, gq = combined["clean"], combined["corrupt"]
    gallery = np.concatenate([gc, gq]); query, test = combined["query"], combined["test"]
    labels = np.tile(source_labels, 2)
    source_ids = np.tile(np.arange(len(source_labels)), 2)
    neighbors, similarities = distinct_neighbors(query, gallery, source_ids, cfg["retrieval"]["neighbors"])
    onehot = np.eye(30)[labels]
    knn, _ = weighted_labels(neighbors, similarities, onehot, cfg["graph"]["edge_temperature"])
    blend = (1 - cfg["retrieval"]["neural_blend_weight"]) * base_query + cfg["retrieval"]["neural_blend_weight"] * knn
    embedding = np.concatenate([gallery, test])
    unary = np.concatenate([onehot, base_test])
    graph, kth, isolated = make_graph(embedding, cfg["graph"])
    propagated, info = diffuse(graph, unary, labels, cfg["graph"])
    independent, readout_info = independent_readout(query, embedding, propagated, base_query, kth, cfg["graph"])
    joint_embedding = np.concatenate([embedding, query])
    joint_unary = np.concatenate([unary, base_query])
    joint_graph, _, _ = make_graph(joint_embedding, cfg["graph"])
    joint_values, joint_info = diffuse(joint_graph, joint_unary, labels, cfg["graph"])
    weight = cfg["graph"]["final_neural_graph_weight"]
    joint = (1 - weight) * base_query + weight * joint_values[-len(query):]
    predicted = dict(baseline=base_query, nearest_neighbor=knn, nearest_neighbor_blend=blend,
                     graph_independent=independent, graph_joint_transductive=joint)
    details = dict(graph_nodes=len(embedding), graph_nonzero_edges=int(graph.nnz),
                   source_images=len(source_labels), source_view_nodes=len(labels),
                   diffusion=info, readout=readout_info, joint_diffusion=joint_info)
    return predicted, details, labels[neighbors], propagated[len(labels):]


def metric(y, p):
    return float(f1_score(y, p.argmax(1), labels=list(range(30)), average="macro", zero_division=0))


def evaluate(y, predictions, prior, reference, iterations):
    return {name: dict(raw_macro_f1=metric(y, probabilities(p)),
                       balanced_macro_f1=metric(y, reference.balance(p, prior, iterations)),
                       accuracy=float((p.argmax(1) == y).mean())) for name, p in predictions.items()}


def seed_diagnostics(y, p, agreement, threshold):
    selected = (p.max(1) >= threshold) & agreement
    return dict(selected=int(selected.sum()), errors=int((p[selected].argmax(1) != y[selected]).sum()),
                precision=float((p[selected].argmax(1) == y[selected]).mean()) if selected.any() else None,
                represented_classes=int(len(np.unique(p[selected].argmax(1)))))


def bootstrap(y, baseline_labels, candidate_labels, metadata, cfg):
    rng = np.random.RandomState(cfg["validation"]["bootstrap_seed"])
    strata = []
    assert (metadata.groupby("group_id").label.nunique() == 1).all()
    for _, frame in metadata.groupby("label"):
        strata.append([g.index.to_numpy() for _, g in frame.groupby("group_id")])
    differences = []
    for _ in range(cfg["validation"]["bootstrap_replicates"]):
        ix = np.concatenate([np.concatenate([groups[i] for i in rng.randint(len(groups), size=len(groups))]) for groups in strata])
        baseline = f1_score(y[ix], baseline_labels[ix], labels=list(range(30)), average="macro", zero_division=0)
        candidate = f1_score(y[ix], candidate_labels[ix], labels=list(range(30)), average="macro", zero_division=0)
        differences.append(candidate - baseline)
    return np.percentile(differences, [2.5, 97.5]).tolist()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=ROOT / "configs/LOCAL-PROP-004-PILOT.yaml")
    cfg = yaml.safe_load(ap.parse_args().config.read_text())
    out = ROOT / cfg["output_root"]; out.mkdir(parents=True, exist_ok=True)
    with (out / "runner.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        started = time.monotonic()
        replay.write_json(out / "status.json", dict(status="preflight", pid=os.getpid(), at=replay.now()))
        torch.set_num_threads(cfg["features"]["cpu_threads"])
        original = json.loads((ROOT / cfg["reference_summary"]).read_text())
        assert original["all_requested_checks_passed"]
        versions = {p: importlib.metadata.version(p) for p in replay.VERSIONS}
        assert versions == original["versions"]
        team = Path(cfg["team_root"])
        train = pd.read_csv(ROOT / cfg["dataset_root"] / "train.csv")
        folds = np.load(team / "cache/scene_folds.npy")
        metadata = pd.read_csv(team / "cache/scene_folds.csv")
        assert np.array_equal(metadata.id, train.id) and np.array_equal(metadata.label, train.label)
        assert np.array_equal(folds, metadata.fold) and (metadata.groupby("group_id").fold.nunique() == 1).all()
        hashes = {}
        for rel, expected in original["provenance"]["input_sha256"].items():
            if rel.startswith("runs/") and "model.pt" not in rel and "cfg.json" not in rel and "probs/" not in rel:
                continue
            path = team / rel
            hashes[str(path)] = replay.digest(path)
            assert hashes[str(path)] == expected, f"Frozen baseline input changed: {rel}"
        source = ROOT / cfg["reference_root"] / "reference_sources"
        for name in ("train.py", "dq.py", "evalres.py"):
            hashes[str(source / name)] = replay.digest(source / name)
            assert hashes[str(source / name)] == original["provenance"]["input_sha256"][name]
        identity = dict(config=cfg, runner_sha256=replay.digest(Path(__file__)), reference_loader_sha256=replay.digest(Path(replay.__file__)),
                        versions=versions, input_sha256=hashes)
        specification = out / "specification.json"
        if specification.exists() and json.loads(specification.read_text()) != identity:
            raise ValueError("Frozen pilot specification or inputs changed")
        replay.write_json(specification, identity)
        reference = replay.load_reference(source)
        feature_checks = {}
        all_predictions, all_y, all_indices, fold_results = {}, [], [], []
        prior = np.bincount(train.label, minlength=30) / len(train)
        for fold in cfg["folds"]:
            ti, vi = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
            assert not (set(metadata.iloc[ti].group_id) & set(metadata.iloc[vi].group_id))
            members = [member_features(prepare_features(cfg, out, ti, vi, fold, member, reference, feature_checks)) for member in cfg["members"]]
            combined = {key: unit(np.concatenate([m[key] for m in members], axis=1)) for key in ("clean", "corrupt", "query", "test")}
            base = {kind: [np.load(ROOT / cfg["reference_root"] / "predictions" / f"{m['run']}_f{fold}" / f"{kind}.npy") for m in cfg["members"]] for kind in ("validation", "test")}
            replay.write_json(out / "status.json", dict(status="graph_inference", fold=fold, pid=os.getpid(), at=replay.now()))
            predicted, details, neighbor_labels, test_predicted = solve_fold(cfg, combined, train.label.to_numpy()[ti],
                                                        np.mean(base["validation"], 0), np.mean(base["test"], 0))
            y = train.label.to_numpy()[vi]
            scores = evaluate(y, predicted, prior, reference, cfg["validation"]["balance_iterations"])
            agreement = base["validation"][0].argmax(1) == base["validation"][1].argmax(1)
            seeds = {name: seed_diagnostics(y, p, agreement & (p.argmax(1) == base["validation"][0].argmax(1)),
                                           cfg["validation"]["seed_confidence_for_diagnosis_only"]) for name, p in predicted.items() if name in ("baseline", "graph_independent")}
            result = dict(fold=fold, rows=len(vi), scores=scores, nearest_neighbor_top1_purity=float((neighbor_labels[:, 0] == y).mean()),
                           nearest_neighbor_top5_mean_purity=float((neighbor_labels[:, :5] == y[:, None]).mean()),
                           seeds=seeds, graph=details, test_labels_changed_from_fold_baseline=int((test_predicted.argmax(1) != np.mean(base["test"], 0).argmax(1)).sum()))
            fold_results.append(result)
            save_npz(out / f"fold{fold}_predictions.npz", validation_indices=vi, **predicted, test_graph=test_predicted)
            for name, p in predicted.items(): all_predictions.setdefault(name, []).append(p)
            all_y.append(y); all_indices.append(vi)
            replay.write_json(out / f"fold{fold}_result.json", result)
            replay.write_json(out / "feature_probability_checks.json", feature_checks)
            replay.log(f"Fold {fold} scores: {json.dumps(scores)}")
        y = np.concatenate(all_y); indices = np.concatenate(all_indices)
        predictions = {name: np.concatenate(parts) for name, parts in all_predictions.items()}
        scores = evaluate(y, predictions, prior, reference, cfg["validation"]["balance_iterations"])
        delta = scores["graph_independent"]["balanced_macro_f1"] - scores["baseline"]["balanced_macro_f1"]
        baseline_labels = reference.balance(predictions["baseline"], prior).argmax(1)
        candidate_labels = reference.balance(predictions["graph_independent"], prior).argmax(1)
        replay.write_json(out / "status.json", dict(status="paired_group_bootstrap", pid=os.getpid(), at=replay.now()))
        interval = bootstrap(y, baseline_labels, candidate_labels, metadata.iloc[indices].reset_index(drop=True), cfg)
        passed = delta >= cfg["validation"]["minimum_primary_pooled_macro_f1_gain"] and all(
            f["scores"]["graph_independent"]["balanced_macro_f1"] - f["scores"]["baseline"]["balanced_macro_f1"] >= cfg["validation"]["minimum_primary_each_fold_macro_f1_gain"] for f in fold_results)
        for path, expected in hashes.items():
            assert replay.digest(path) == expected, f"Input changed during pilot {path}"
        result = dict(pilot_id=cfg["pilot_id"], at=replay.now(), duration_seconds=time.monotonic()-started,
                      validation_rows=len(y), folds=fold_results, pooled_scores=scores,
                      primary_pooled_delta=delta, paired_group_bootstrap_delta_95_percent=interval,
                      changed_validation_predictions=int((candidate_labels != baseline_labels).sum()),
                      fixed_errors=int(((baseline_labels != y) & (candidate_labels == y)).sum()),
                      broken_predictions=int(((baseline_labels == y) & (candidate_labels != y)).sum()),
                      graph_component_screen_passed=bool(passed), source_inputs_unchanged=True,
                      training_performed=False, submission_performed=False, geometric_matching_performed=False,
                      full_five_fold_oof=False, primary_protocol=cfg["validation"]["primary"], secondary_protocol=cfg["validation"]["secondary"])
        replay.write_json(out / "result.json", result)
        replay.write_json(out / "status.json", dict(status="completed", graph_component_screen_passed=bool(passed), at=replay.now()))
        replay.log(json.dumps(result))


if __name__ == "__main__":
    main()
