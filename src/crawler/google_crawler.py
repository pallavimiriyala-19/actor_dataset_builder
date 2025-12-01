import os
import requests
from duckduckgo_search.duckduckgo_search import DDGS

def crawl_actor_images(actor_name, raw_dir, max_images=50):
    os.makedirs(raw_dir, exist_ok=True)

    query = f"{actor_name} actor face closeup portrait HD"
    print(f"[CRAWL] DuckDuckGo → {query}")

    ddgs = DDGS()   # correct class for your version

    try:
        results = ddgs.images(
            keywords=query,
            max_results=max_images
        )
    except Exception as e:
        print("[ERROR] DuckDuckGo failed:", e)
        return

    if not results:
        print("[WARN] No images found!")
        return

    count = 0

    for i, item in enumerate(results):
        url = item.get("image")
        if not url:
            continue

        try:
            img_data = requests.get(url, timeout=10).content
            file_path = os.path.join(raw_dir, f"{actor_name}_{i}.jpg")

            with open(file_path, "wb") as f:
                f.write(img_data)

            count += 1

        except Exception:
            continue

    print(f"[INFO] Downloaded {count} images.")
