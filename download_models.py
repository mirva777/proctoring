"""
download_models.py
==================
Downloads the MediaPipe Tasks tflite model files needed by the pipeline.

Run once before first use:
    python download_models.py
"""
import urllib.request
from pathlib import Path

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

MODELS = {
    "blaze_face_short_range.tflite": (
        "https://storage.googleapis.com/mediapipe-models/"
        "face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
    ),
    "face_landmarker.task": (
        "https://storage.googleapis.com/mediapipe-models/"
        "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    ),
}


def download(name: str, url: str) -> None:
    dest = MODELS_DIR / name
    if dest.exists():
        print(f"  [skip] {name} already downloaded")
        return
    print(f"  Downloading {name} ...", end=" ", flush=True)
    urllib.request.urlretrieve(url, dest)
    size_mb = dest.stat().st_size / 1_048_576
    print(f"done ({size_mb:.1f} MB)")


if __name__ == "__main__":
    print("Downloading MediaPipe model files to ./models/\n")
    for name, url in MODELS.items():
        download(name, url)
    print("\nAll models ready.")
