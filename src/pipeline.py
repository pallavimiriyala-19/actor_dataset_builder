import os
import json
import shutil
import numpy as np

from src.config import Config
from src.crawler.google_crawler import crawl_actor_images
from src.crawler.imdb_scraper import scrape_imdb_images
from src.crawler.wikipedia_scraper import fetch_wikipedia_anchors
from src.processing.face_detector import detect_and_crop_faces
from src.processing.duplicate_filter import remove_duplicates
from src.processing.identity_filter import filter_by_identity
from src.embedding.embedder import build_embeddings


def normalize(name: str) -> str:
    return name.strip().replace(" ", "_")


def build_actor_dataset(actor: str, cfg: Config | None = None, wiki_title: str | None = None) -> dict:
    cfg = cfg or Config()
    key = normalize(actor)
    raw_dir = os.path.join(cfg.raw_root, key)
    person_root = os.path.join(cfg.people_root, key)
    face_dir = os.path.join(person_root, "images")
    anchor_dir = os.path.join(person_root, "_anchor")

    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(face_dir, exist_ok=True)
    os.makedirs(anchor_dir, exist_ok=True)

    print(f"\n=== {actor} ===")

    print("[ANCHOR] Wikipedia reference photo(s)...")
    anchor_paths = fetch_wikipedia_anchors(actor, anchor_dir, wiki_title=wiki_title)

    print("[CRAWL] DDGS + Bing + Google image search...")
    crawl_actor_images(actor, raw_dir, cfg.max_google_images)

    print("[CRAWL] IMDb...")
    try:
        scrape_imdb_images(actor, raw_dir, cfg.max_imdb_images)
    except Exception as e:
        print("[IMDb] skipped:", e)

    print("[FACE] Detect & crop...")
    detect_and_crop_faces(
        raw_dir,
        face_dir,
        cfg.min_face_size,
        cfg.face_confidence_threshold,
    )

    print("[DEDUP] Removing near-duplicates...")
    remove_duplicates(face_dir)

    print("[IDENT] Identity verification...")
    kept, removed, used_anchor, anchor_source = filter_by_identity(
        face_dir,
        anchor_paths,
        sim_threshold=cfg.identity_sim_threshold,
        refine_threshold=cfg.identity_refine_threshold,
    )

    print("[EMBED] Building embeddings...")
    embeddings, mean_emb = build_embeddings(face_dir)

    if embeddings is not None:
        np.save(os.path.join(person_root, "embeddings.npy"), embeddings)
        np.save(os.path.join(person_root, "mean_embedding.npy"), mean_emb)

    raw_count = len([f for f in os.listdir(raw_dir) if os.path.isfile(os.path.join(raw_dir, f))])
    clean_count = len([f for f in os.listdir(face_dir) if os.path.isfile(os.path.join(face_dir, f))])

    meta = {
        "name": actor,
        "images_raw": raw_count,
        "images_clean": clean_count,
        "identity_removed": removed,
        "anchor_source": anchor_source,
        "embedding_shape": list(embeddings.shape) if embeddings is not None else None,
    }
    with open(os.path.join(person_root, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # Drop the anchor scratch dir to keep `people/<actor>/` clean.
    try:
        shutil.rmtree(anchor_dir)
    except Exception:
        pass

    # Free disk: cleaned face crops are now in `people/<actor>/images/` and
    # raw downloads are no longer needed.
    if cfg.delete_raw_after_processing and clean_count > 0:
        try:
            shutil.rmtree(raw_dir)
        except Exception as e:
            print(f"[CLEANUP] could not remove {raw_dir}: {e}")

    print(f"[DONE] {actor}: raw={raw_count} clean={clean_count} "
          f"id_removed={removed} emb={meta['embedding_shape']}")
    return meta
