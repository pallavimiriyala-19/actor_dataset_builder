from dataclasses import dataclass

@dataclass
class Config:
    raw_root: str = "raw_data"
    people_root: str = "people"

    max_google_images: int = 40
    max_imdb_images: int = 10

    min_face_size: int = 120
    face_confidence_threshold: float = 0.40
