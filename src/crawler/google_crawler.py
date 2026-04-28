import os
import time
import requests
from ddgs import DDGS

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

MIN_BYTES = 5 * 1024
IMAGE_MAGIC = (b"\xff\xd8\xff", b"\x89PNG\r\n", b"GIF87a", b"GIF89a", b"RIFF", b"BM")


def _looks_like_image(blob: bytes) -> bool:
    if len(blob) < MIN_BYTES:
        return False
    return any(blob.startswith(sig) for sig in IMAGE_MAGIC)


def _fetch(url: str, timeout: int = 12) -> bytes | None:
    try:
        r = requests.get(url, headers=HEADERS, timeout=timeout, stream=False)
        if r.status_code != 200:
            return None
        if "image" not in r.headers.get("Content-Type", "").lower() and not _looks_like_image(r.content):
            return None
        return r.content
    except Exception:
        return None


def _ddgs_search(query: str, max_results: int) -> list[str]:
    urls: list[str] = []
    try:
        ddgs = DDGS()
        results = ddgs.images(query=query, max_results=max_results) or []
        for item in results:
            u = item.get("image") or item.get("thumbnail")
            if u:
                urls.append(u)
    except Exception as e:
        print(f"[WARN] DDGS failed for '{query}':", e)
    return urls


def _icrawler_search(engine: str, query: str, max_results: int, scratch_dir: str) -> list[str]:
    """Use icrawler (Bing or Google) as an additional source. Returns saved file paths."""
    try:
        if engine == "bing":
            from icrawler.builtin import BingImageCrawler as Crawler
        elif engine == "google":
            from icrawler.builtin import GoogleImageCrawler as Crawler
        else:
            return []
    except Exception as e:
        print(f"[WARN] icrawler {engine} not available:", e)
        return []
    os.makedirs(scratch_dir, exist_ok=True)
    try:
        crawler = Crawler(storage={"root_dir": scratch_dir}, log_level=40)
        crawler.crawl(keyword=query, max_num=max_results, file_idx_offset=0)
    except Exception as e:
        print(f"[WARN] {engine} crawler failed for '{query}':", e)
    return [
        os.path.join(scratch_dir, f)
        for f in os.listdir(scratch_dir)
        if os.path.isfile(os.path.join(scratch_dir, f))
    ]


def _bing_search(query: str, max_results: int, scratch_dir: str) -> list[str]:
    return _icrawler_search("bing", query, max_results, scratch_dir)


def _google_search(query: str, max_results: int, scratch_dir: str) -> list[str]:
    return _icrawler_search("google", query, max_results, scratch_dir)


def crawl_actor_images(actor_name: str, raw_dir: str, max_images: int = 100) -> int:
    """Download up to `max_images` candidate photos for `actor_name` into `raw_dir`.

    Strategy: try DDGS image search first with several query variants; fall back to
    Bing via icrawler. Returns the count of files actually saved.
    """
    os.makedirs(raw_dir, exist_ok=True)

    queries = [
        f"{actor_name} actor face closeup portrait HD",
        f"{actor_name} actor headshot",
        f"{actor_name} Telugu actor photo",
        f"{actor_name} actor",
    ]

    seen_urls: set[str] = set()
    candidates: list[str] = []
    for q in queries:
        if len(candidates) >= max_images * 2:
            break
        for u in _ddgs_search(q, max_results=max_images):
            if u not in seen_urls:
                seen_urls.add(u)
                candidates.append(u)
        time.sleep(0.5)

    print(f"[CRAWL] DDGS candidates: {len(candidates)} for '{actor_name}'")

    saved = 0
    safe = actor_name.replace(" ", "_")
    for i, url in enumerate(candidates):
        if saved >= max_images:
            break
        blob = _fetch(url)
        if not blob:
            continue
        ext = ".jpg"
        path = os.path.join(raw_dir, f"{safe}_ddg_{i:04d}{ext}")
        try:
            with open(path, "wb") as f:
                f.write(blob)
            saved += 1
        except Exception:
            continue

    # Top up with Bing and Google via icrawler
    for engine, label in (("bing", "bing"), ("google", "google")):
        if saved >= max_images:
            break
        remaining = max_images - saved
        scratch = os.path.join(raw_dir, f"_{label}_scratch")
        files = _icrawler_search(engine, f"{actor_name} actor", remaining, scratch)
        for j, src in enumerate(files):
            if saved >= max_images:
                break
            try:
                with open(src, "rb") as f:
                    blob = f.read()
                if not _looks_like_image(blob):
                    continue
                dst = os.path.join(raw_dir, f"{safe}_{label}_{j:04d}.jpg")
                with open(dst, "wb") as f:
                    f.write(blob)
                saved += 1
            except Exception:
                continue
        try:
            for f in os.listdir(scratch):
                os.remove(os.path.join(scratch, f))
            os.rmdir(scratch)
        except Exception:
            pass

    print(f"[CRAWL] Saved {saved} raw images for '{actor_name}' → {raw_dir}")
    return saved
