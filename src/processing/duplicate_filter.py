from PIL import Image
import os, imagehash

def remove_duplicates(folder):
    hashes = {}
    for f in os.listdir(folder):
        path = os.path.join(folder, f)
        try:
            img = Image.open(path)
            h = imagehash.average_hash(img)
        except:
            continue

        if h in hashes:
            os.remove(path)
        else:
            hashes[h] = f
