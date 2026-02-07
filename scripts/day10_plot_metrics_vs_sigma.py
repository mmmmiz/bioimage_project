from __future__ import annotations
import argparse
import re
from dataclasses import dataclass
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

def _to_gray(img: np.ndarray) -> np.ndarray:
    """
    img: OpenCVで読み込んだ画像（2次元=グレー or 3次元=カラー(BGR)）
    return: 2次元のグレースケール(uint8想定)
    """
    if img.ndim == 2:
        gray = img
    elif img.ndim == 3 and img.shape[2] == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif img.ndim == 3 and img.shape[2] == 4:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError(f"想定外の画像shape: {img.shape}")
    
    if gray.dtype != np.uint8:
        gray = np.clip(gray, 0 , 255).astype(np.uint8)
    return gray

def calc_brightness(gray: np.ndarray) -> float:
    return float(np.mean(gray))

def calc_contrast(gray: np.ndarray)->float:
    return float(np.std(gray))

def calc_sharpness(gray: np.ndarray)->float:
    # Laplacianの分散（ピンぼけほど小さくなる傾向）
    lap = cv2.Laplacian(gray, ddepth=cv2.CV_64F)
    return float(np.var(lap))

@dataclass(frozen=True)
class Row:
    sigma: float
    image_path:str
    brightness_mean:float
    contrast_std:float
    sharpness_lap_var:float
    
def _iter_sigma_dirs(root:Path)->list[tuple[float,Path]]:
    """
    root配下から sigma_0 / sigma_1.5 のようなディレクトリを拾う
    """
    sigma_dirs :list[tuple[float,Path]] = []
    pat = re.compile(r"^sigma_(\d+(?:\.\d+)?)$") # フォルダ名の作成ルール（sigma_1,sigma_1.5）
    
    for p in sorted(root.iterdir()):
        if not p.is_dir():
            continue
        m = pat.match(p.name)
        if not m:
            continue
        sigma = float(m.group(1))
        sigma_dirs.append((sigma, p))
        
    if not sigma_dirs:
        raise FileNotFoundError(f"{root} 配下に 'sigma_数字' ディレクトリが見つかりませんでした。例: sigma_0, sigma_1, sigma_2"
        )
    return sigma_dirs

def _iter_images(dir_path:Path)->list[Path]:
    imgs = [p for p in sorted(dir_path.rglob("*")) if p.suffix.lower() in IMG_EXTS]
    return imgs

def build_rows(root:Path)->list[Row]:
    rows:list[Row] = []
    for sigma,d in _iter_sigma_dirs(root):
        images = _iter_images(d)
        if not images:
            print(f"[WARN] {d} に画像がありません。スキップします。")
            continue
        for img_path in images:
            img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
            if img is None:
                print(f"[WORN]読み込み失敗:{img_path}")
                continue
            
            gray = _to_gray(img)
            rows.append(
                Row(
                    sigma=sigma,
                    image_path=str(img_path),
                    brightness_mean=calc_brightness(gray),
                    contrast_std=calc_contrast(gray),
                    sharpness_lap_var=calc_sharpness(gray)
                )
            )
            
    if not rows:
        raise RuntimeError("有効な画像が1枚も処理できませんでした。")
    return rows

def save_plots(df_mean:pd.DataFrame,out_dir:Path)->None:
    """
    df_mean: sigmaごとの平均（列に brightness_mean / contrast_std / sharpness_lap_var がある想定）
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 1図1指標で保存（見返しやすい）
    for col in ["brightness_mean","contrast_std", "sharpness_lap_var"]:
        plt.figure()
        plt.plot(df_mean["sigma"], df_mean[col], marker="o")
        plt.xlabel("sigma")
        plt.ylabel(col)
        plt.title(f"{col} vs sigma")
        plt.tight_layout()
        plt.savefig(out_dir / f"{col}_vs_sigma.png", dpi=150)
        plt.close()
    
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=str, help="sigma_* ディレクトリ群が入っているルートフォルダ")
    ap.add_argument("--out", type=str, default="outputs/day10", help="出力先ディレクトリ")
    return ap.parse_args()

def main()-> int:
    args = parse_args()
    root = Path(args.root)
    out_dir = Path(args.out)
    
    rows = build_rows(root)
    df = pd.DataFrame([r.__dict__ for r in rows])
    
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "metrics_all.csv", index=False)
    
    # sigmaごとの平均
    df_mean = (
        df.groupby("sigma", as_index=False)[
            ["brightness_mean", "contrast_std", "sharpness_lap_var"]
        ]
        .mean()
        .sort_values("sigma")
    )
    df_mean.to_csv(out_dir / "metrics_mean_by_sigma.csv", index=False)
    save_plots(df_mean,out_dir)
    
    print("== Saved ==")
    print(f"- {out_dir / 'metrics_all.csv'}")
    print(f"- {out_dir / 'metrics_mean_by_sigma.csv'}")
    print(f"- {out_dir / 'brightness_mean_vs_sigma.png'} (etc...)")
    print()
    print("== Mean by sigma ==")
    print(df_mean.to_string(index=False))

    return 0
if __name__ == "__main__":
    raise SystemExit(main())