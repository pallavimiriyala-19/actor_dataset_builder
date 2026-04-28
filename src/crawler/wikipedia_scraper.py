"""Fetch a high-confidence reference photo for an actor from Wikipedia/Wikimedia.

Used as the identity anchor for filtering wrong-person photos out of the dataset.
Tries multiple disambiguation variants so common names (e.g. "Sunil") resolve to
the actor, not an unrelated namesake.
"""
import os
import requests

UA = {"User-Agent": "actor-dataset-builder/1.0 (research)"}

# Tokens that suggest the page is about an actor / film personality.
ACTOR_HINTS = (
    "actor", "actress", "film", "cinema", "telugu", "tollywood",
    "bollywood", "kollywood", "filmmaker", "director",
)

# Tokens that say "this is a disambiguation page".
DISAMBIG_HINTS = ("may refer to", "disambiguation")


def _summary(title: str) -> dict | None:
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{title.replace(' ', '_')}"
    try:
        r = requests.get(url, headers=UA, timeout=10)
        if r.status_code != 200:
            return None
        return r.json()
    except Exception:
        return None


def _looks_like_actor(d: dict) -> bool:
    text = " ".join(
        str(d.get(k, "")) for k in ("description", "extract")
    ).lower()
    if any(h in text for h in DISAMBIG_HINTS):
        return False
    return any(h in text for h in ACTOR_HINTS)


def _resolve_title(name: str, hint_title: str | None) -> dict | None:
    """Try several disambiguation variants and return the first that looks like an actor."""
    candidates = []
    if hint_title:
        candidates.append(hint_title)

    candidates.extend([
        name,
        f"{name} (actor)",
        f"{name} (Telugu actor)",
        f"{name} (Indian actor)",
        f"{name} (Telugu film actor)",
        f"{name} (film actor)",
    ])

    seen = set()
    for cand in candidates:
        if cand in seen:
            continue
        seen.add(cand)
        d = _summary(cand)
        if not d:
            continue
        if _looks_like_actor(d):
            return d
        # If the very first hit is plausibly the right person and isn't a
        # disambiguation page, accept as a last-resort fallback later.

    # Final fallback: return the plain-name summary even if hints don't match,
    # so we at least try something rather than skip the anchor entirely.
    return _summary(name)


def _media_list(title: str, max_items: int = 4) -> list[str]:
    url = f"https://en.wikipedia.org/api/rest_v1/page/media-list/{title.replace(' ', '_')}"
    try:
        r = requests.get(url, headers=UA, timeout=10)
        if r.status_code != 200:
            return []
        items = r.json().get("items", [])
    except Exception:
        return []
    urls = []
    for it in items:
        if it.get("type") != "image":
            continue
        src = (it.get("srcset") or [{}])[0].get("src")
        if src:
            if src.startswith("//"):
                src = "https:" + src
            urls.append(src)
        if len(urls) >= max_items:
            break
    return urls


def fetch_wikipedia_anchors(
    actor_name: str,
    save_dir: str,
    max_extra: int = 3,
    wiki_title: str | None = None,
) -> list[str]:
    """Save Wikipedia lead photo + a few additional page images. Returns saved paths."""
    os.makedirs(save_dir, exist_ok=True)
    saved: list[str] = []

    summary = _resolve_title(actor_name, wiki_title)
    if not summary:
        print(f"[WIKI] {actor_name}: no Wikipedia entry")
        return []

    candidates: list[str] = []
    if "originalimage" in summary:
        candidates.append(summary["originalimage"]["source"])
    elif "thumbnail" in summary:
        candidates.append(summary["thumbnail"]["source"])

    page_title = summary.get("titles", {}).get("canonical") or summary.get("title")
    if page_title:
        for u in _media_list(page_title, max_items=max_extra):
            if u not in candidates:
                candidates.append(u)

    seen = set()
    for i, url in enumerate(candidates):
        if url in seen:
            continue
        seen.add(url)
        try:
            r = requests.get(url, headers=UA, timeout=15)
            if r.status_code != 200 or len(r.content) < 5 * 1024:
                continue
            ext = ".png" if url.lower().endswith(".png") else ".jpg"
            path = os.path.join(save_dir, f"_wiki_{i:02d}{ext}")
            with open(path, "wb") as f:
                f.write(r.content)
            saved.append(path)
        except Exception:
            continue

    print(f"[WIKI] {actor_name}: {len(saved)} reference photo(s) (page='{page_title}')")
    return saved
