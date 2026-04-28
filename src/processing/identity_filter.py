"""Identity-verification filter: removes wrong-person crops from an actor's folder.

Strategy:
1. Build an *anchor* embedding from Wikipedia reference photo(s) when available.
2. Embed every candidate crop.
3. If we have an anchor: keep crops with cosine_sim(crop, anchor) >= threshold.
4. If no anchor: cluster candidates and keep the dominant cluster (the actor will
   normally be the most-photographed identity in the search results).
5. Refine: recompute anchor as the mean of kept embeddings and re-threshold once.

This is the single most important step for face-recognition data quality —
without it, co-stars and lookalikes contaminate the actor's folder.
"""
from __future__ import annotations

import os
import cv2
import numpy as np

from src.embedding.embedder import _get_app


def _l2norm(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    n = np.where(n == 0, 1.0, n)
    return x / n


def _embed_image(app, path: str) -> np.ndarray | None:
    img = cv2.imread(path)
    if img is None:
        return None
    try:
        faces = app.get(img)
    except Exception:
        return None
    if len(faces) != 1:
        return None
    return faces[0].embedding.astype(np.float32)


def _build_anchor(app, anchor_paths: list[str]) -> np.ndarray | None:
    embs = []
    for p in anchor_paths:
        e = _embed_image(app, p)
        if e is not None:
            embs.append(e)
    if not embs:
        return None
    a = np.stack(embs).mean(axis=0)
    return _l2norm(a[None, :])[0]


def _dominant_cluster(embs: np.ndarray, eps: float = 0.45, min_samples: int = 3) -> np.ndarray:
    """DBSCAN on cosine distance; return boolean mask for the dominant cluster.

    eps=0.45 cosine ≈ angle ~58° works well for ArcFace embeddings to group the
    same identity while still separating different people.
    """
    try:
        from sklearn.cluster import DBSCAN
    except Exception:
        # No sklearn — keep everything (no filter).
        return np.ones(len(embs), dtype=bool)

    if len(embs) < min_samples:
        return np.ones(len(embs), dtype=bool)

    normed = _l2norm(embs)
    cos_dist = 1.0 - normed @ normed.T
    cos_dist = np.clip(cos_dist, 0.0, 2.0)

    db = DBSCAN(eps=eps, min_samples=min_samples, metric="precomputed")
    labels = db.fit_predict(cos_dist)

    if (labels >= 0).sum() == 0:
        # No cluster reached min_samples — keep nothing rather than guess.
        return np.zeros(len(embs), dtype=bool)

    # Pick the cluster with the most members.
    valid = labels[labels >= 0]
    counts = np.bincount(valid)
    dominant = int(np.argmax(counts))
    return labels == dominant


def filter_by_identity(
    face_dir: str,
    anchor_paths: list[str],
    sim_threshold: float = 0.42,
    refine_threshold: float = 0.50,
) -> tuple[int, int, np.ndarray | None, str]:
    """Remove crops in `face_dir` whose embedding doesn't match the actor's identity.

    Returns (kept, removed, anchor_used). `anchor_used` is the L2-normalized
    embedding actually applied for filtering (after refinement), useful as the
    actor's gallery embedding.
    """
    app = _get_app()

    files = [
        f for f in sorted(os.listdir(face_dir))
        if os.path.isfile(os.path.join(face_dir, f))
    ]
    if not files:
        return 0, 0, None, "none"

    embs: list[np.ndarray] = []
    paths: list[str] = []
    for f in files:
        p = os.path.join(face_dir, f)
        e = _embed_image(app, p)
        if e is None:
            os.remove(p)
            continue
        embs.append(e)
        paths.append(p)

    if not embs:
        return 0, len(files), None, "none"

    E = np.stack(embs)
    En = _l2norm(E)

    anchor = _build_anchor(app, anchor_paths)
    if anchor is not None:
        anchor_source = "wikipedia"
        sims = En @ anchor
        keep_mask = sims >= sim_threshold
        # If the anchor is poor (very few matches), fall back to clustering.
        if keep_mask.sum() < max(3, int(0.10 * len(En))):
            keep_mask = _dominant_cluster(E)
            anchor = None  # will be rebuilt below
            anchor_source = "cluster"
    else:
        keep_mask = _dominant_cluster(E)
        anchor_source = "cluster"

    # Refinement: rebuild anchor from kept set, re-threshold with stricter cutoff.
    if keep_mask.sum() >= 3:
        refined_anchor = _l2norm(En[keep_mask].mean(axis=0)[None, :])[0]
        sims = En @ refined_anchor
        keep_mask = sims >= refine_threshold
        anchor = refined_anchor

    removed = 0
    for i, p in enumerate(paths):
        if not keep_mask[i]:
            try:
                os.remove(p)
                removed += 1
            except FileNotFoundError:
                pass

    kept = int(keep_mask.sum())
    print(f"[IDENT] {face_dir}: kept {kept}/{len(paths)} (removed {removed} wrong-identity crops)")
    return kept, removed, anchor, anchor_source
