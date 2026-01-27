import sys
from pathlib import Path
import cv2

def main()->int:
    # print(sys.argv)すると['scripts/check_image.py', 'data/sample.jpg']と出る。[スクリプト名,画像パス]
    if len(sys.argv) < 2:
        print("Usage: python scripts/check_image.py <image_path>")
        return 2
    image_path = Path(sys.argv[1])
    
    if not image_path.exists():
        print(f"[ERROR] File not found: {image_path}")
        return 2
    img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    # cv2.imread(...) は「画像ファイル → 画像データ（NumPy配列）」に変換する関数
    # cv2.IMREAD_UNCHANGED は「元の形式をなるべくそのまま読み込む」指定
    if img is None:
        print("[ERROR] cv2.imread failed. Unsupported format or broken file.")
        return 1
    print(f"path :{image_path}")
    print(f"shape :{img.shape}")
    print(f"path :{img.dtype}")
    
    return 0

if __name__ == "__main__":
    raise SystemExit(main())