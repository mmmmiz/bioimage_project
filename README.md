# Bioimage Quality Check App  （画像品質チェックアプリ）
細胞などの生物画像（bioimage）を対象に、画像の品質（見やすさ・ピンぼけ等）を数値化して判定し、結果をUIで確認・CSV出力できる ローカル実行アプリを作るプロジェクトです。UIは Streamlit、実装言語は Python を使用し、深層学習（PyTorch / TensorFlow）は使わない方針です。
## 目的
- 画像を読み込み、品質指標を計算して OK/NG を判定する
- しきい値（判定基準）を調整できるUIを用意する
- 判定結果を CSVでダウンロードできるようにする
技術スタック / 実行環境

## 実行環境：ローカル（Mac）
- 言語：Python
- UI：Streamlit
- 利用ライブラリ：NumPy / OpenCV / pandas / matplotlib（検証・可視化）

## Metrics（指標）
3つの指標で画像品質を数値化します:
- **brightness_mean**: 平均輝度（明るさ）
- **contrast_std**: コントラスト（メリハリ）
- **sharpness_lap_var**: シャープネス（ピントの鮮明さ）

詳細 → [docs/metrics.md](docs/metrics.md)

## Judgement（判定）
各指標にしきい値を設定し、範囲外ならNG + 理由を列挙します。

詳細 → [docs/judgement.md](docs/judgement.md)

このアプリは 明るさ・コントラスト・シャープネス の3指標で画像品質をチェックし、しきい値に基づいて OK/NG を判定します。