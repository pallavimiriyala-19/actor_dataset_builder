import os, cv2
from insightface.app import FaceAnalysis

face_app = FaceAnalysis(name="buffalo_l")
face_app.prepare(ctx_id=0)

def detect_and_crop_faces(raw_dir, face_dir, min_size, threshold):
    os.makedirs(face_dir, exist_ok=True)
    cropped = []

    for file in os.listdir(raw_dir):
        path = os.path.join(raw_dir, file)
        img = cv2.imread(path)
        if img is None: continue

        faces = face_app.get(img)
        if len(faces) != 1: continue

        face = faces[0]
        x1, y1, x2, y2 = map(int, face.bbox)
        w = x2 - x1

        if w < min_size or face.det_score < threshold:
            continue

        crop = img[y1:y2, x1:x2]
        save_path = os.path.join(face_dir, file)
        cv2.imwrite(save_path, crop)
        cropped.append(save_path)

    return cropped
