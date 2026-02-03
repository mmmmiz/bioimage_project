# Day7: 指標計算をここに集約（カラー/グレー両対応）

from __future__ import annotations
import numpy as np
import cv2

from .io import to_gray
from .exceptions import ImageQualityError

def _as_float(gray: np.ndarray) ->np.ndarray:
    """計算用にfloatへ。"""
    if gray.size == 0:
        raise ImageQualityError("画像が空です")
    return gray.astype(np.float32, copy=False)

def cal_brightness(img: np.ndarray) -> float:
    gray = _as_float(to_gray(img))
    return float(np.mean(gray))

def cal_contrast(img: np.ndarray) -> float:
    gray = _as_float(to_gray(img))
    return float(np.std(gray))

def cal_sharpness(img: np.ndarray) -> float:
    gray = to_gray(img)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return float(lap.var())