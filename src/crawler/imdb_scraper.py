import os
import requests

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
}


def scrape_imdb_images(actor_name: str, save_dir: str, max_num: int = 20) -> int:
    """Best-effort IMDb photo scrape using cinemagoer. Silently skips on any failure."""
    try:
        from imdb import Cinemagoer
    except Exception as e:
        print("[IMDb] cinemagoer not installed, skipping:", e)
        return 0

    os.makedirs(save_dir, exist_ok=True)

    try:
        ia = Cinemagoer()
        results = ia.search_person(actor_name)
    except Exception as e:
        print(f"[IMDb] search failed for {actor_name}: {e}")
        return 0

    if not results:
        print(f"[IMDb] no results for {actor_name}")
        return 0

    try:
        person = ia.get_person(results[0].personID)
    except Exception as e:
        print(f"[IMDb] get_person failed for {actor_name}: {e}")
        return 0

    photos = person.get("photos") or person.get("headshot")
    if isinstance(photos, str):
        photos = [photos]
    if not photos:
        print(f"[IMDb] no photos for {actor_name}")
        return 0

    saved = 0
    safe = actor_name.replace(" ", "_")
    for i, p in enumerate(photos[:max_num]):
        if isinstance(p, dict):
            url = p.get("url") or p.get("full-size url") or p.get("large")
        else:
            url = p
        if not url:
            continue
        try:
            r = requests.get(url, headers=HEADERS, timeout=12)
            if r.status_code != 200 or len(r.content) < 5 * 1024:
                continue
            with open(os.path.join(save_dir, f"{safe}_imdb_{i:03d}.jpg"), "wb") as f:
                f.write(r.content)
            saved += 1
        except Exception:
            continue

    print(f"[IMDb] saved {saved} images for {actor_name}")
    return saved
