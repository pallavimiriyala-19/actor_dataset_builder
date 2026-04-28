import os
import cv2

from src.embedding.embedder import _get_app


def detect_and_crop_faces(raw_dir: str, face_dir: str, min_size: int = 120, threshold: float = 0.40, margin: float = 0.25):
    os.makedirs(face_dir, exist_ok=True)
    app = _get_app()
    cropped = []

    files = sorted(os.listdir(raw_dir))
    for file in files:
        path = os.path.join(raw_dir, file)
        if not os.path.isfile(path):
            continue
        img = cv2.imread(path)
        if img is None:
            continue

        try:
            faces = app.get(img)
        except Exception:
            continue
        if len(faces) != 1:
            continue

        face = faces[0]
        x1, y1, x2, y2 = map(int, face.bbox)
        w = x2 - x1
        h = y2 - y1

        if w < min_size or h < min_size:
            continue
        if float(face.det_score) < threshold:
            continue

        # add margin around bbox so we don't cut hair / chin
        mx, my = int(w * margin), int(h * margin)
        H, W = img.shape[:2]
        xx1 = max(0, x1 - mx)
        yy1 = max(0, y1 - my)
        xx2 = min(W, x2 + mx)
        yy2 = min(H, y2 + my)

        crop = img[yy1:yy2, xx1:xx2]
        if crop.size == 0:
            continue

        save_path = os.path.join(face_dir, os.path.splitext(file)[0] + ".jpg")
        cv2.imwrite(save_path, crop)
        cropped.append(save_path)

    file_count = sum(1 for f in files if os.path.isfile(os.path.join(raw_dir, f)))
    print(f"[FACE] kept {len(cropped)}/{file_count} from {raw_dir}")
    return cropped
