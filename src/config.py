import os
from dataclasses import dataclass


@dataclass
class Config:
    raw_root: str = "raw_data"
    people_root: str = "people"

    max_google_images: int = 100
    max_imdb_images: int = 20

    min_face_size: int = 100
    face_confidence_threshold: float = 0.50

    # Identity verification thresholds (cosine similarity on L2-normalized
    # ArcFace embeddings, range -1..1).
    # First-pass cutoff against the Wikipedia anchor — permissive so we don't
    # lose hard angles / lighting before refinement.
    identity_sim_threshold: float = 0.42
    # Second-pass cutoff against the refined anchor (mean of first-pass keeps).
    identity_refine_threshold: float = 0.50

    # Delete raw_data/<actor>/ after the cleaned face crops + embeddings have
    # been saved. Saves significant disk: raw images are typically 3-4x the
    # size of the verified crop set.
    delete_raw_after_processing: bool = True


def configure_threading():
    """Speed up CPU inference by using all cores. Safe to call once at startup."""
    cpu = os.cpu_count() or 4
    os.environ.setdefault("OMP_NUM_THREADS", str(cpu))
    os.environ.setdefault("MKL_NUM_THREADS", str(cpu))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(cpu))
