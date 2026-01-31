# Day6: 指標計算フロー統合
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import cv2
import numpy as np

def read_image_bgr(path:Path)-> np.ndarray:
    """OpenCVで画像をBGRとして読み込む（読み込めない場合は例外）"""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"画像を読み込めませんでした: {path}")
    return img

def to_gray(img_bgr: np.ndarray)-> np.ndarray:
    if img_bgr.ndim != 3 or img_bgr.shape[2] != 3:
        raise ValueError(f"BGR画像(高さ,幅,3)を想定していますが、shape={img_bgr.shape} でした")
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

def calc_brightness(gray: np.ndarray)->float:
    """平均輝度（明るさ）"""
    return float(np.mean(gray))

def calc_contrast(gray: np.ndarray)->float:
    """標準偏差（コントラスト）"""
    return float(np.std(gray))

def calc_sharpness(gray: np.ndarray)->float:
    """Laplacian分散（シャープネス）"""
    gray_f = gray.astype(np.float64)
    lap = cv2.Laplacian(gray_f, ddepth=cv2.CV_64F, ksize=3)
    return float(np.var(lap))

def evaluate_image(path:Path)->dict[str,float]:
    """読み込み→変換→指標計算の一連処理をまとめる"""
    img = read_image_bgr(path)
    gray = to_gray(img)
    
    return {
        "brightness_mean": calc_brightness(gray),
        "contrast_std": calc_contrast(gray),
        "sharpness_laplacian_var": calc_sharpness(gray),
    }
    
def parse_args(argv:list[str])-> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate image quality metrics for a single image.")
    parser.add_argument("image", type=Path, help="評価したい画像ファイルへのパス")
    return parser.parse_args(argv)

def main(argv:list[str] | None = None)->int:
    args = parse_args(argv if argv is not None else sys.argv[1:])
    
    try:
        metrics = evaluate_image(args.image)
    except Exception as e:
        print(f"[ERROR]{e}", file = sys.stderr)
        return 1
    
    print("== Metrics ==")
    for k, v in metrics.items():
        print(f"{k}:{v:.4f}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

"""
IMREAD_COLOR は「常に 3 チャンネルの BGR に変換して読み込む」フラグ（グレースケール画像もBGRになる）

argparse がコマンドライン引数を解析した結果を、「名前付きで入れておく箱」みたいなオブジェクト。parser.parse_args() の戻り値としてよく出てくる。args.image のように ドットで取り出せる。

metrics.item() 
辞書の中身を「(key, value) の組」で見せる“見え方”

sys.argv[1:]
sys.argv  python scripts/eval_image.py data/sample.jpgの python以降
[1:]  「1番目から最後まで」
python scripts/eval_image.py data/sample.jpg
のなかの『data/sample.jpg』
"""
