# Day7: 合成画像で最低限の動作確認（画像ファイル不要）

# Day13: 統合パイプラインの自己テスト
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR / "src"))

from bioimage_qc.pipeline import evaluate, evaluate_image, compute_metrics

EXPECTED_METRIC_KEYS = {"brightness_mean", "contrast_std", "sharpness_lap_var"}
EXPECTED_TOP_KEYS = {"input_path", "metrics", "judgement", "meta"}


def test_evaluate_legacy():
    """Day7: 既存 evaluate() が壊れていないことを確認。"""
    gray = np.zeros((10, 10), dtype=np.uint8)
    r = evaluate(gray)
    assert set(r.keys()) == {"brightness", "contrast", "sharpness"}
    assert all(isinstance(v, float) for v in r.values())
    print("[PASS] test_evaluate_legacy")


def test_compute_metrics():
    """compute_metrics のキー名が標準形に合っていること。"""
    img = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
    m = compute_metrics(img)
    assert set(m.keys()) == EXPECTED_METRIC_KEYS
    assert all(isinstance(v, float) for v in m.values())
    print("[PASS] test_compute_metrics")


def test_evaluate_image_with_tempfile():
    """evaluate_image が結果dictの標準スキーマを返すこと。"""
    import cv2
    import tempfile

    # 一時画像を作成
    img = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        cv2.imwrite(f.name, img)
        tmp_path = Path(f.name)

    result = evaluate_image(tmp_path)
    assert set(result.keys()) == EXPECTED_TOP_KEYS
    assert set(result["metrics"].keys()) == EXPECTED_METRIC_KEYS
    assert isinstance(result["judgement"]["ok"], bool)
    assert result["judgement"]["label"] in ("OK", "NG")
    assert isinstance(result["judgement"]["reasons"], list)

    tmp_path.unlink()  # 後片付け
    print("[PASS] test_evaluate_image_with_tempfile")


def test_judge_ng_case():
    """sharpness しきい値を極端に高くして NG になることを確認。"""
    import cv2
    import tempfile

    # ぼやけた画像（全ピクセル同値 → sharpness ≈ 0）
    img = np.full((50, 50, 3), 128, dtype=np.uint8)
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        cv2.imwrite(f.name, img)
        tmp_path = Path(f.name)

    thresholds = {"sharpness_lap_var": {"min": 99999.0}}
    result = evaluate_image(tmp_path, thresholds=thresholds)

    assert result["judgement"]["ok"] is False
    assert result["judgement"]["label"] == "NG"
    assert len(result["judgement"]["reasons"]) > 0

    tmp_path.unlink()
    print("[PASS] test_judge_ng_case")


def main() -> int:
    test_evaluate_legacy()
    test_compute_metrics()
    test_evaluate_image_with_tempfile()
    test_judge_ng_case()
    print("\n=== All tests passed ===")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

"""
python scripts/self_test.py
全4テストが通れば、以下が表示されるはずです：


[PASS] test_evaluate_legacy
[PASS] test_compute_metrics
[PASS] test_evaluate_image_with_tempfile
[PASS] test_judge_ng_case

=== All tests passed ===
各テストの確認内容：

テスト	                 何を確認しているか
test_evaluate_legacy	Day7の evaluate() が壊れていないこと
test_compute_metrics	compute_metrics() のキー名が標準形 (sharpness_lap_var 等) と一致すること

test_evaluate_image_with_tempfile	evaluate_image() が結果dictの全キー（input_path, metrics, judgement, meta）を返すこと

test_judge_ng_case	しきい値を極端に高くしたとき、ちゃんと NG になること
"""
