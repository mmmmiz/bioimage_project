# NEW: しきい値設計（全画像の指標集計→しきい値候補を算出）

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class Item:
    path: Path
    label: str  # "sharp" or "blur"
    sigma: float  # blurならsigma、sharpは0.0扱い


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", default="data/dataset", help="data/dataset を指定")
    p.add_argument("--out-csv", default="data/analysis/metrics_all.csv", help="集計CSVの出力先")
    p.add_argument("--out-json", default="data/analysis/thresholds.json", help="しきい値候補JSONの出力先")
    p.add_argument("--imread", choices=["color", "gray", "unchanged"], default="color")
    return p.parse_args()


def _iter_items(dataset_root: Path) -> list[Item]:
    """
    data/dataset/
      sharp/...
      blur/sigma_1/... などを走査して Item のリストにする
    """
    items: list[Item] = []

    sharp_dir = dataset_root / "sharp"
    if sharp_dir.exists():
        for p in sorted(sharp_dir.rglob("*")):
            if p.is_file() and p.suffix.lower() in IMG_EXTS:
                items.append(Item(path=p, label="sharp", sigma=0.0))

    blur_dir = dataset_root / "blur"
    pat = re.compile(r"^sigma_(\d+(?:\.\d+)?)$")
    if blur_dir.exists():
        for d in sorted(blur_dir.iterdir()):
            if not d.is_dir():
                continue
            m = pat.match(d.name)
            if not m:
                continue
            sigma = float(m.group(1))
            for p in sorted(d.rglob("*")):
                if p.is_file() and p.suffix.lower() in IMG_EXTS:
                    items.append(Item(path=p, label="blur", sigma=sigma))

    return items


def _read_image(path: Path, imread_mode: str) -> np.ndarray:
    if imread_mode == "color":
        flag = cv2.IMREAD_COLOR
    elif imread_mode == "gray":
        flag = cv2.IMREAD_GRAYSCALE
    else:
        flag = cv2.IMREAD_UNCHANGED

    img = cv2.imread(str(path), flag)
    if img is None:
        raise ValueError(f"failed to read image: {path}")
    return img


def _to_gray(img: np.ndarray) -> np.ndarray:
    # OpenCVのcolorはBGR (Blue, Green, Red)
    if img.ndim == 2:
        return img
    if img.ndim == 3 and img.shape[2] == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 想定外（例：RGBAなど）は、とりあえず先頭3chで灰度化
    if img.ndim == 3 and img.shape[2] >= 3:
        bgr = img[:, :, :3]
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    raise ValueError(f"unsupported image shape: {img.shape}")


def calc_metrics(img: np.ndarray) -> dict[str, float]:
    gray = _to_gray(img)
    gray_f = gray.astype(np.float32)

    brightness_mean = float(np.mean(gray_f))
    contrast_std = float(np.std(gray_f))

    lap = cv2.Laplacian(gray_f, cv2.CV_32F)
    sharpness_lap_var = float(np.var(lap))

    return {
        "brightness_mean": brightness_mean,
        "contrast_std": contrast_std,
        "sharpness_lap_var": sharpness_lap_var,
    }


def _balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # ラベル: sharp=1, blur=0 で扱う
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # sharp recall
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0  # blur recall
    return 0.5 * (tpr + tnr)


def propose_thresholds(df: pd.DataFrame) -> dict[str, object]:
    """
    しきい値候補を提案する（最初はsharpness中心）
    - sharpness: balanced accuracy が最大になる閾値を探索
    - brightness: sharp の 1%〜99% を許容範囲（初期案）
    - contrast: sharp の 1%点を下限（初期案）
    """
    sharp = df[df["label"] == "sharp"]
    blur = df[df["label"] == "blur"]
    if len(sharp) == 0 or len(blur) == 0:
        raise ValueError("sharp と blur の両方が必要です（dataset構成を確認してください）")

    # --- sharpness threshold search ---
    vals = df["sharpness_lap_var"].to_numpy()
    candidates = np.unique(np.quantile(vals, np.linspace(0.05, 0.95, 200)))

    y_true = (df["label"] == "sharp").to_numpy().astype(int)  # sharp=1, blur=0
    best_t = float(candidates[0])
    best_score = -1.0
    for t in candidates:
        y_pred = (df["sharpness_lap_var"].to_numpy() >= t).astype(int)
        score = _balanced_accuracy(y_true, y_pred)
        if score > best_score:
            best_score = score
            best_t = float(t)

    # --- brightness / contrast (guard) ---
    b_min = float(sharp["brightness_mean"].quantile(0.01))
    b_max = float(sharp["brightness_mean"].quantile(0.99))
    c_min = float(sharp["contrast_std"].quantile(0.01))

    return {
        "sharpness_lap_var": {"min": best_t},
        "brightness_mean": {"min": b_min,"max": b_max},
        "contrast_std": {"min": c_min},
        "meta":{
            "sharpness_search_balanced_accuracy": float(best_score),
        },
    }


def main() -> int:
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    out_csv = Path(args.out_csv)
    out_json = Path(args.out_json)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    items = _iter_items(dataset_root)
    if not items:
        raise FileNotFoundError(f"no images found under: {dataset_root}")

    rows: list[dict[str, object]] = []
    for it in items:
        img = _read_image(it.path, args.imread)
        m = calc_metrics(img)
        rows.append(
            {
                "path": str(it.path),
                "label": it.label,
                "sigma": it.sigma,
                **m,
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)

    # 集計表示（まずはここが出ればOK）
    print("== counts ==")
    print(df["label"].value_counts())
    print("\n== by sigma (blur) ==")
    print(df[df["label"] == "blur"].groupby("sigma")["sharpness_lap_var"].describe()[["count", "mean", "min", "50%", "max"]])

    thresholds = propose_thresholds(df)
    out_json.write_text(json.dumps(thresholds, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n== proposed thresholds ==")
    print(json.dumps(thresholds, indent=2, ensure_ascii=False))
    print(f"\nWrote: {out_csv}")
    print(f"Wrote: {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
