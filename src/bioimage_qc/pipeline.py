# Day7: 1枚画像を評価する「動線」

from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from .metrics import cal_brightness, cal_contrast, cal_sharpness

def evaluate(img: np.ndarray) -> dict[str, float]:
    return {
        "brightness": cal_brightness(img),
        "contrast": cal_contrast(img),
        "sharpness": cal_sharpness(img),
    }
