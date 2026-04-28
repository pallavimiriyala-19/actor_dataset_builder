import os
import cv2
import numpy as np
from insightface.app import FaceAnalysis

_app = None


def _get_app():
    global _app
    if _app is None:
        try:
            import onnxruntime as ort
            providers = ort.get_available_providers()
        except Exception:
            providers = ["CPUExecutionProvider"]

        if "CUDAExecutionProvider" in providers:
            chosen = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            ctx_id = 0
        else:
            chosen = ["CPUExecutionProvider"]
            ctx_id = -1

        app = FaceAnalysis(name="buffalo_l", providers=chosen)
        app.prepare(ctx_id=ctx_id, det_size=(640, 640))
        _app = app
    return _app


def build_embeddings(face_folder: str):
    app = _get_app()
    arr = []

    for file in sorted(os.listdir(face_folder)):
        path = os.path.join(face_folder, file)
        if not os.path.isfile(path):
            continue
        img = cv2.imread(path)
        if img is None:
            continue
        try:
            faces = app.get(img)
        except Exception:
            continue
        if len(faces) == 1:
            arr.append(faces[0].embedding)

    if arr:
        embs = np.stack(arr).astype(np.float32)
        return embs, embs.mean(axis=0)
    return None, None
