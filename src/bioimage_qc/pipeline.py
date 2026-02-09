# Day7: 1枚画像を評価する「動線」

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import numpy as np

from .io import read_image, to_gray
from .metrics import cal_brightness, cal_contrast, cal_sharpness
from .judge import JudgeConfig, judge_metrics


def evaluate(img: np.ndarray) -> dict[str, float]:
    return {
        "brightness": cal_brightness(img),
        "contrast": cal_contrast(img),
        "sharpness": cal_sharpness(img),
    }

# ---------- Day13 追加 ----------
"""
metrics のキー名を 
brightness_mean / contrast_std / sharpness_lap_var に揃える（judge.pyの _RULE_ORDER と一致させる）
既存の evaluate() は残す（既存スクリプトの互換性維持）
"""

def compute_metrics(img: np.ndarray) -> dict[str, float]:
    """指標を計算し、結果dictの"metrics"キー名に合わせて返す。"""
    return {
        "brightness_mean": cal_brightness(img),
        "contrast_std": cal_contrast(img),
        "sharpness_lap_var": cal_sharpness(img),
    }


def evaluate_image(
    path: str | Path,
    config: JudgeConfig | None = None,
    imread: str = "color",
) -> dict[str, Any]:
    """
    1枚の画像パスを受け取り、指標計算→判定→結果dictを返す。

    Parameters
    ----------
    path : 画像ファイルパス
    config : 判定しきい値
    imread : "color" / "gray"

    Returns
    -------
    dict : Day13で定めた標準スキーマの結果dict
    """
    import cv2

    p = Path(path)

    # --- 読み込み ---
    flags = cv2.IMREAD_GRAYSCALE if imread == "gray" else cv2.IMREAD_COLOR
    img = read_image(p, flags=flags)

    # --- 指標計算 ---
    metrics = compute_metrics(img)

    # --- 判定 ---
    if config is None:
        config = JudgeConfig()
    result = judge_metrics(metrics, config)

    # --- 結果dict化 ---
    return {
        "input_path": str(p),
        "metrics": metrics,
        "judgement": {
            "ok": result.verdict.value == "OK",
            "label": result.verdict.value,
            "reasons": list(result.reasons),
        },
        "meta": {
            "imread": imread,
            "notes": "",
        },
    }
    