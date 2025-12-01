import os, json, numpy as np
from src.config import Config
from src.crawler.google_crawler import crawl_actor_images
from src.crawler.imdb_scraper import scrape_imdb_images
from src.processing.face_detector import detect_and_crop_faces
from src.processing.duplicate_filter import remove_duplicates
from src.embedding.embedder import build_embeddings

def normalize(name):
    return name.strip().replace(" ", "_")

def build_actor_dataset(actor, cfg=Config()):
    key = normalize(actor)
    raw_dir = os.path.join(cfg.raw_root, key)
    face_dir = os.path.join(cfg.people_root, key, "images")
    person_root = os.path.join(cfg.people_root, key)

    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(face_dir, exist_ok=True)

    print("[CRAWL] Google/Bing:", actor)
    crawl_actor_images(actor, raw_dir, cfg.max_google_images)

    print("[CRAWL] IMDb:", actor)
    scrape_imdb_images(actor, raw_dir, cfg.max_imdb_images)

    print("[FACE] Detect & Crop...")
    detect_and_crop_faces(raw_dir, face_dir, cfg.min_face_size, cfg.face_confidence_threshold)

    print("[DEDUP] Removing duplicates...")
    remove_duplicates(face_dir)

    print("[EMBED] Building embeddings...")
    embeddings, mean_emb = build_embeddings(face_dir)

    if embeddings is not None:
        np.save(os.path.join(person_root, "embeddings.npy"), embeddings)

    meta = {
        "name": actor,
        "images_raw": len(os.listdir(raw_dir)),
        "images_clean": len(os.listdir(face_dir)),
        "embedding_shape": list(embeddings.shape) if embeddings is not None else None
    }

    with open(os.path.join(person_root, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("✓ DONE:", actor)
