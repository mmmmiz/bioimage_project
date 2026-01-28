# CLI demo(画像一枚⇨明るさ表示)
from __future__ import annotations
from pathlib import Path
import argparse
import cv2 
from metrics import calc_brightness

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image",required=True,help="Path to image file")
    args = parser.parse_args()
    
    image_path = Path(args.image)
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        raise FileNotFoundError(f"Failed to read image: {image_path}")
    b = calc_brightness(img_bgr)
    print (f"brightness(mean gray): {b:.2f}")
    return 0

if __name__ == "__main__":
    main()