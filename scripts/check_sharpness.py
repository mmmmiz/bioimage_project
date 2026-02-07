from __future__ import annotations
import argparse
import cv2
from scripts.metrics_demo import calc_sharpness

def main() ->int:
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", help="チェックしたい画像のパス")
    args = parser.parse_args()
    
    img_bgr = cv2.imread(args.image_path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise SystemExit(f"画像が読み込めません: {args.image_path}")
    score = calc_sharpness(img_bgr)
    print(f"sharpness_lap_var={score:.2f}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
