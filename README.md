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