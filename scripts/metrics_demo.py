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

def calc_contrast(img_gray)-> float:
    """
    グレースケール画像のコントラストを計算する
    コントラスト = 画素値の標準偏差
    """
    return float(np.std(img_gray))
    #入力はグレースケール前提、戻り値は float に統一
    
def _to_gray(img: np.ndarray) -> np.ndarray:
    """BGR/Grayどちらでも受け取り、Gray(2次元)に揃える。"""
    if img.ndim == 2:
        return img
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    raise ValueError(f"Unsupported image shape: {img.shape}")

def calc_sharpness(img:np.ndarray) -> float:
    """
    シャープネス（ピンぼけ）指標：Variance of Laplacian
    - 大きいほどシャープ、小さいほどボケ傾向
    """
    gray = _to_gray(img)
    lap = cv2.Laplacian(gray, ddepth=cv2.CV_64F, ksize=3)
    
    score = float(np.var(lap))
    return score
