import os, cv2, numpy as np
from insightface.app import FaceAnalysis

app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=0)

def build_embeddings(face_folder):
    arr = []

    for file in os.listdir(face_folder):
        path = os.path.join(face_folder, file)
        img = cv2.imread(path)
        if img is None: continue

        faces = app.get(img)
        if len(faces) == 1:
            arr.append(faces[0].embedding)

    if arr:
        embs = np.stack(arr)
        return embs, embs.mean(axis=0)
    
    return None, None
