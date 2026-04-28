# Actor Dataset Builder

Crawls web + IMDb images for an actor, detects and crops a single face per image,
removes near-duplicates, and produces face embeddings (InsightFace `buffalo_l`).

## 1. System dependencies

```bash
sudo apt update
sudo apt install build-essential python3-dev ffmpeg libgl1 -y
```

## 2. Python environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

If you have an NVIDIA GPU, swap `onnxruntime` for `onnxruntime-gpu` in
`requirements.txt`. The code auto-detects CUDA and falls back to CPU.

> If your IDE shows "package not installed" hints on `requirements.txt`, point
> the IDE's Python interpreter at `./venv/bin/python`.

## 3. Run

Single actor:

```bash
python -m src.cli.main --actor "Mahesh Babu" --max-images 100
```

Multiple actors:

```bash
python -m src.cli.main --actor "Pawan Kalyan" "Ram Charan" --max-images 80
```

Batch from file (one name per line, `#` lines ignored):

```bash
python -m src.cli.main --list telugu_actors.txt --max-images 80
```

A starter list of Telugu film industry actors is provided in
[`telugu_actors.txt`](telugu_actors.txt).

## Output layout

```
raw_data/<Actor_Name>/                 # raw downloaded photos
people/<Actor_Name>/
    images/                            # cleaned, single-face crops
    embeddings.npy                     # (N, 512) face embeddings
    mean_embedding.npy                 # (512,) mean embedding
    metadata.json                      # counts + embedding shape
people/_summary.json                   # per-actor summary across the run
```

## Tunables (`src/config.py`)

| field                       | default | meaning                                |
|-----------------------------|---------|----------------------------------------|
| `max_google_images`         | 100     | max web images per actor               |
| `max_imdb_images`           | 20      | max IMDb photos per actor              |
| `min_face_size`             | 100     | minimum face bbox width/height (px)    |
| `face_confidence_threshold` | 0.50    | InsightFace detector score threshold   |
