# scripts/batch_eval.py
# Day19: フォルダ内画像を一括評価してCSV出力（新規）
from __future__ import annotations
import sys
from pathlib import Path
import argparse
import csv
import json

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from bioimage_qc.pipeline import evaluate_image

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("input_dir", type=str, help="画像フォルダ（再帰的に探索）")
    p.add_argument("--out", type=str, default="outputs/results.csv", help="出力CSVパス")
    p.add_argument("--thresholds",type=str, default="", help="しきい値JSON（任意）例: configs/thresholds.json")
    return p.parse_args()

def load_thresholds(path_str: str) -> dict | None:
    if not path_str:
        return None
    p = Path(path_str)
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def iter_images(root: Path) -> list[Path]:
    return [p for p in sorted(root.rglob("*")) if p.is_file() and p.suffix.lower() in IMG_EXTS]


def flatten_result(result: dict, thresholds_used: dict | None) -> dict:
    # metrics
    metrics = result.get("metrics", {})
    # judgement
    judgement = result.get("judgement", {})
    reasons = judgement.get("reasons", [])
    if isinstance(reasons, list):
        reasons_str = "; ".join(map(str, reasons))
    else:
        reasons_str = str(reasons)

    row = {
        "input_path": str(result.get("input_path", "")),
        "ok": bool(judgement.get("ok", False)),
        "label": str(judgement.get("label", "OK" if judgement.get("ok", False) else "NG")),
        "reasons": reasons_str,
        # metrics columns
        "brightness_mean": float(metrics.get("brightness_mean", float("nan"))),
        "contrast_std": float(metrics.get("contrast_std", float("nan"))),
        "sharpness_lap_var": float(metrics.get("sharpness_lap_var", float("nan"))),
        # thresholds（記録用：JSON文字列にして1列で保持）
        "thresholds_json": json.dumps(thresholds_used, ensure_ascii=False) if thresholds_used is not None else "",
    }
    return row

def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"input_dir not found: {input_dir}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    thresholds = load_thresholds(args.thresholds)

    images = iter_images(input_dir)
    if not images:
        raise ValueError(f"no images found under: {input_dir}")

    fieldnames = [
        "input_path",
        "ok",
        "label",
        "reasons",
        "brightness_mean",
        "contrast_std",
        "sharpness_lap_var",
        "thresholds_json",
    ]

    rows: list[dict] = []
    for img_path in images:
        # ★ 重要: pipelineがthresholds対応しているなら渡す
        result = evaluate_image(img_path, thresholds=thresholds) if thresholds is not None else evaluate_image(img_path)

        # input_path が入っていない実装もあるので、ここで補う
        if not result.get("input_path"):
            result["input_path"] = str(img_path)

        row = flatten_result(result, thresholds)
        rows.append(row)

    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"wrote: {out_path} (rows={len(rows)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
