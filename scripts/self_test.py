# Day7: 合成画像で最低限の動作確認（画像ファイル不要）

from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR / "src"))

from bioimage_qc.pipeline import evaluate

def main() -> int:
    gray = np.zeros((10, 10), dtype=np.uint8)
    color = np.zeros((10, 10, 3), dtype=np.uint8)
    
    r1 = evaluate(gray)
    r2 = evaluate(color)
    
    print("gray:", r1)
    print("color:", r2)
    
    assert set(r1.keys()) == {"brightness", "contrast", "sharpness"}
    assert all(isinstance(v, float) for v in r1.values())
    assert set(r2.keys()) == set(r1.keys())
    return 0

if __name__ == "__main__":
    raise SystemExit(main())