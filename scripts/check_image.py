from __future__ import annotations
import sys
from pathlib import Path
import argparse

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR / "src"))

from bioimage_qc.io import read_image
from bioimage_qc.pipeline import evaluate
from bioimage_qc.exceptions import ImageQualityError

def main()->int:
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path",help="画像パス（例： data/sample.jpg)")
    args = parser.parse_args()
    
    try:
        img = read_image(args.image_path)
        result = evaluate(img)
    except ImageQualityError as e:
        print(f"[ERROR] {e}")
        return 2
    for k, v in result.items():
        print(f"{k}: {v:.3f}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())