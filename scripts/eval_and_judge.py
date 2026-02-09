"""Day13: 1枚画像の評価・判定CLI"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR / "src"))

from bioimage_qc.pipeline import evaluate_image
from bioimage_qc.judge import JudgeConfig, Range


def main() -> int:
    parser = argparse.ArgumentParser(description="1枚画像を評価・判定する")
    parser.add_argument("image", type=Path, help="画像ファイルパス")
    parser.add_argument("--imread", default="color", choices=["color", "gray"])
    parser.add_argument("--sharpness-min", type=float, default=None,
    help="sharpness の下限しきい値（例: 500）")
    args = parser.parse_args()

    # しきい値のカスタマイズ（指定があれば）
    config = None
    if args.sharpness_min is not None:
        config = JudgeConfig(
            sharpness_lap_var=Range(min=args.sharpness_min, max=None),
        )

    result = evaluate_image(args.image, config=config, imread=args.imread)

    # 結果表示
    print(f"File : {result['input_path']}")
    print(f"Label: {result['judgement']['label']}")
    print(f"Metrics:")
    for k, v in result["metrics"].items():
        print(f"  {k}: {v:.2f}")
    if result["judgement"]["reasons"]:
        print("Reasons:")
        for r in result["judgement"]["reasons"]:
            print(f"  - {r}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""
python scripts/eval_and_judge.py data/raw/sample.jpg --sharpness-min 500

①何をしたか:
sample.jpg という1枚の画像に対して、Day13の統合パイプラインを実行。

python scripts/eval_and_judge.py data/raw/sample.jpg --sharpness-min 500
このコマンドは内部で以下の処理を 1本の流れ で行っています：

画像の読み込み — data/raw/sample.jpg をカラーで読み込み
指標計算 — 明るさ・コントラスト・シャープネスの3指標を算出
判定 — 各指標がしきい値の範囲内かを判定
結果dict化 — 結果をまとめて表示
結果の読み方
指標	値	意味
brightness_mean	143.71	平均輝度。0〜255の中間あたりで、明るさは普通
contrast_std	42.37	輝度の標準偏差。コントラストもそこそこある
sharpness_lap_var	213.69	ラプラシアン分散。画像のシャープさ（ピント具合）

②なぜ NG になったか
--sharpness-min 500 を指定したので、「シャープネスが 500以上 ならOK」というしきい値が設定されました。

しかし実際のシャープネスは 213.69 で、500に届いていません。

そのため
Label: NG
Reason: sharpness_lap_var=213.688 は許容範囲[500, ∞]から外れています
つまり「この画像はしきい値500の基準で見ると ピントが甘い（ぼやけている） 」という判定結果です。

--sharpness-min を指定しなければデフォルト（0以上＝何でもOK）になるので、同じ画像でも OK になります。"""