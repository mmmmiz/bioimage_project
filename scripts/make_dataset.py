# Day9 データセット作成（sharp/blur 生成 + manifest.csv 出力）

from __future__ import annotations
import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
import cv2

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

@dataclass(frozen=True)
class Record:
    src: str
    dst: str
    label: str
    sigma: float
    ksize: int
    
def _parse_sigmas(text:str) ->list[float]:
    """
    "1,2,3" のような文字列を [1.0, 2.0, 3.0] にする
    """
    if not text.strip():
        return[]
    sigmas: list[float] = []
    for part in text.split(","):
        s = float(part.strip())
        if s < 0:
            raise ValueError(f"sigma は0以上にしてください: {s}")
        sigmas.append(s)
    return sigmas

def _sigma_dirname(sigma:float) -> str:
    """
    0.5 -> "sigma_0p5" のようにフォルダ名に安全な形へ
    """
    s = f"{sigma:g}"
    return "sigma_" + s.replace(".","p")

def _auto_ksize(sigma: float) -> int:
    """
    設計上の決め（OpenCV内部の自動計算と一致させる必要はない）
    sigma が大きいほどカーネルを大きくし、必ず奇数にする。
    """
    k = int(round(sigma * 6 + 1)) # “だいたい 6σ 幅”という目安
    k = max(k, 3)
    if k % 2 == 0:
        k += 1
    return k

def _iter_images(input_path:Path) ->list[Path]:
    """
    "1,2,3" のような文字列を [1.0, 2.0, 3.0] にする
    """
    if input_path.is_file():
        return[input_path]
    if input_path.is_dir():
        files = [p for p in input_path.rglob("*") if p.suffix.lower() in IMG_EXTS]
        return sorted(files)
    
    raise FileNotFoundError(f"入力が見つかりません: {input_path}")

def _imread_any(path: Path):
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"画像として読み込めません: {path}")
    return img
def _write_image(dst: Path, img) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(dst), img)
    if not ok:
        raise RuntimeError(f"画像の保存に失敗しました: {dst}")

def main() -> int:
    parser = argparse.ArgumentParser(description="Day9: blur dataset generator")
    parser.add_argument("input", type=str, help="入力画像ファイル or 入力フォルダ（再帰で探索）")
    parser.add_argument("--out", type=str, default="data/dataset",help="出力先ルートフォルダ")
    parser.add_argument("--sigmas", type=str, default="1,2,3",help='ぼかし強度σ（例: "1,2,3"）')
    args = parser.parse_args()
    
    input_path = Path(args.input)
    out_root = Path(args.out)
    sigmas = _parse_sigmas(args.sigmas)
    
    sharp_dir = out_root / "sharp"
    blur_root = out_root / "blur"
    sharp_dir.mkdir(parents=True, exist_ok=True)
    blur_root.mkdir(parents=True, exist_ok=True)
    
    images =_iter_images(input_path)
    if not images:
        raise ValueError(f"入力に画像がありません: {input_path}")
    records: list[Record] = []
    
    for src_path in images:
        img = _imread_any(src_path)
        
        # sharp（元画像コピー）
        sharp_dst = sharp_dir / src_path.name
        _write_image(sharp_dst, img)
        records.append(
            Record(src=str(src_path), dst=str(sharp_dst), label="sharp", sigma=0.0, ksize=0)
        )
        
        # blur（σごとに生成）
        for sigma in sigmas:
            if sigma == 0:
                # sigma=0は元画像コピー
                sigma_dir = blur_root / _sigma_dirname(sigma)
                blur_dst = sigma_dir / src_path.name
                _write_image(blur_dst, img)
                records.append(
                    Record(src=str(src_path), dst=str(blur_dst), label="blur", sigma=0.0, ksize=0)
                )
            else:
                # sigma > 0 の場合はGaussianBlurを適用
                k = _auto_ksize(sigma)
                blurred = cv2.GaussianBlur(img, (k,k), sigmaX=sigma, sigmaY=sigma)
                sigma_dir = blur_root / _sigma_dirname(sigma)
                blur_dst = sigma_dir / src_path.name
                _write_image(blur_dst, blurred)
                records.append(
                    Record(src=str(src_path), dst=str(blur_dst), label="blur", sigma=sigma, ksize=k)
                )
    manifest = out_root / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["src", "dst", "label", "sigma", "ksize"])
        for r in records:
            w.writerow([r.src, r.dst, r.label, r.sigma, r.ksize])
        
    print(f"done: {len(records)} records => manifest: {manifest}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
