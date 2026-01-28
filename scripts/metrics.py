from __future__ import annotations
from typing import Union
import cv2 
import numpy as np

def calc_brightness(img: np.ndarray) -> float:
    """
    明るさ（平均輝度）を返す。
    - 入力: グレースケール(H, W) または BGRカラー(H, W, 3) のNumPy配列
    - 出力: 平均輝度（float）
    """
    if img is None:
        raise ValueError("img is None")
    if img.ndim == 2:
        gray = img
    elif img.ndim == 3 and img.shape[2] == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError(f"Unsupported image shape: {img.shape}")
    brightness = float(np.mean(gray, dtype=np.float64))
    return brightness

