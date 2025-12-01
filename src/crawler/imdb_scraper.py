from imdb import IMDb
import requests, os

def scrape_imdb_images(actor_name, save_dir, max_num=10):
    ia = IMDb()
    results = ia.search_person(actor_name)

    if not results:
        print(f"[WARN] No IMDb results for {actor_name}")
        return
    
    actor = ia.get_person(results[0].personID)
    photos = actor.get("photos", [])[:max_num]

    for i, p in enumerate(photos):
        try:
            img = requests.get(p["url"]).content
            with open(os.path.join(save_dir, f"imdb_{i}.jpg"), "wb") as f:
                f.write(img)
        except:
            pass
