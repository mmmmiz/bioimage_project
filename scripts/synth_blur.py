# Day8: 人工ピンぼけ生成（GaussianBlur）を作成

from __future__ import annotations
import argparse
from pathlib import Path
import cv2
import numpy as np

def _sigma_to_ksize(sigma: float)-> tuple[int, int]:
    """sigma からカーネルサイズ(奇数)を作る簡易ルール。
    目安: k ≈ 6*sigma + 1（四捨五入）→ 奇数に丸める
    """
    if sigma <= 0:
        return (1, 1)
    k = int(round(sigma * 6 + 1))
    if k % 2 == 0:
        k += 1
    k = max(k, 3)
    return (k, k)
def synth_blur(img: np.ndarray, sigma:float, ksize: tuple[int, int] | None = None) -> np.ndarray:
    """
    画像をGaussianBlurでぼかす。sigma<=0ならコピーを返す。
    - img: (H,W) or (H,W,3) を想定（グレー/カラー両対応）
    """
    if img is None:
        raise ValueError("img in None")
    if sigma <= 0:
        return img.copy()
    if ksize is None:
        ksize = _sigma_to_ksize(sigma)
    if ksize[0] <= 0 or ksize[1] <=0 or ksize[0] % 2 == 0 or ksize[1] % 2 == 0:
        raise ValueError(f"ksize must be positive odd. got ksize={ksize}")
    blurred = cv2.GaussianBlur(img, ksize, sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_DEFAULT)
    return blurred

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate synthetic defocus blur using GaussianBlur.")
    p.add_argument("input", type=str, help="Input image path")
    p.add_argument("--sigma", type=float, default=2.0, help="Blur strength (standard deviation). 0 means no blur.")
    p.add_argument("--out", type=str, default="", help="Output image path (default: alongside input)")
    p.add_argument("--ksize", type=int, default=0, help="Kernel size (odd). 0 means auto from sigma.")
    p.add_argument(
        "--imread",
        choices=["unchanged", "color", "gray"],
        default="unchanged",
        help="How to read the image",
    )
    return p.parse_args(argv)

def main() -> int:
    args = parse_args()
    in_path = Path(args.input)
    if not in_path.exists():
        raise FileNotFoundError(f"input not found: {in_path}")
    
    if args.imread == "color":
        flag = cv2.IMREAD_COLOR
    elif args.imread == "gray":
        flag = cv2.IMREAD_GRAYSCALE
    else :
        flag = cv2.IMREAD_UNCHANGED
        
    img = cv2.imread(str(in_path), flag)
    if img is None:
        raise ValueError(f"failed to read image: {in_path}")
    
    sigma = float(args.sigma)
    if sigma < 0:
        raise ValueError("--sigma must be >= 0")
    
    ksize = None
    if args.ksize !=0:
        if args.ksize <= 0 or args.ksize % 2 == 0:
            raise ValueError("--ksize must be a positive odd integer (or 0 for auto).")
        ksize = (args.ksize, args.ksize)
    
    out_path = Path(args.out) if args.out else in_path.with_name(f"{in_path.stem}_sigma{sigma:g}{in_path.suffix}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    blurred = synth_blur(img, sigma=sigma, ksize=ksize)
    
    ok = cv2.imwrite(str(out_path), blurred)
    if not ok:
        raise ValueError(f"failed to write image: {out_path}")
    
    print("== synth_blur ==")
    print(f"input : {in_path}")
    print(f"shape : {img.shape}, dtype: {img.dtype}")
    print(f"sigma : {sigma}")
    print(f"ksize : {ksize if ksize is not None else _sigma_to_ksize(sigma)}")
    print(f"output: {out_path}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

"""
ksize を自動で決める簡単ルール例：
ksize ≈ 6*sigma + 1 を四捨五入して 奇数にする
例：sigma→ksize は 0→1, 0.5→5, 1→7, 2→13, 3→19, 4→25

kは最低3以上にしておく（sigmaが小さくても安定）
"""