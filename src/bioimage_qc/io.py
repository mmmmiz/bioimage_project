# Day7: 読み込みとグレースケール変換の一本化

from __future__ import annotations
import cv2
import numpy as np
from pathlib import Path
from .exceptions import ImageQualityError

def read_image(path: str |Path, *, flags: int = cv2.IMREAD_UNCHANGED) -> np.ndarray:
    """画像を読み込んで ndarray を返す。読めなければ ImageQualityError."""
    p = Path(path)
    if not p.exists():
        raise ImageQualityError(f"画像ファイルが見つかりません: {p}")
    img = cv2.imread(str(p), flags)
    if img is None:
        raise ImageQualityError(f"画像を読み込めませんでした形式/権限/破損など）: {p}")
    return img

def to_gray(img: np.ndarray) ->np.ndarray:
    """(H,W) or (H,W,C) を受けて必ず (H,W) のグレー画像にする。"""
    if not isinstance(img, np.ndarray):
        raise ImageQualityError(f"img は numpy.ndarray を期待しました: {type(img)}")
    if img.ndim == 2:
        return img  # 既にグレーならそのまま返す
    if img.ndim == 3:
        h,w,c = img.shape
        if c == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if c == 4:
            return cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        raise ImageQualityError(f"未知のチャンネル数です（想定:3or4）: shape={img.shape}")
    raise ImageQualityError(f"未知の次元数です（想定:2or3）: shape={img.ndim}")
