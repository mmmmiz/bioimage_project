# Bioimage Quality Check App

生物画像（bioimage）の品質を、古典的な画像処理指標で評価して OK/NG 判定するローカル実行アプリです。  
UI は Streamlit、コア処理は Python + NumPy/OpenCV で実装しています。

## 背景

顕微鏡画像などの観察では、暗すぎる・コントラスト不足・ピンぼけが解析精度や目視確認に影響します。  
本プロジェクトでは、深層学習を使わずに次の3指標で品質を定量化し、しきい値ベースで判定できるようにしました。

## 目的

- 画像1枚を読み込み、品質指標を計算する
- しきい値にもとづいて OK/NG を判定する
- 判定理由を表示・CSV保存できるようにする

## スコープ

- ローカル実行（Mac想定）
- 深層学習は不使用
- 単純かつ説明可能なしきい値判定

## 技術スタック

- Python
- Streamlit
- NumPy
- OpenCV
- pandas
- matplotlib（検証用プロット）

## 品質評価の方法

### 指標（Metrics）詳細

#### 1. `brightness_mean`（平均輝度）

何を測るか:
画像全体がどれくらい明るいか（グレースケール 0〜255 の平均値）

計算方法:
`np.mean(gray)`

解釈:
- 高い: 全体が明るい（白っぽい）
- 低い: 全体が暗い（黒っぽい）

注意:
白飛びして情報が潰れていても値は高くなる

#### 2. `contrast_std`（コントラスト）

何を測るか:
明暗のメリハリの強さ

計算方法:
`np.std(gray)`

解釈:
- 高い: 明暗差が大きい
- 低い: のっぺりした画像

注意:
ノイズが強いと標準偏差が上がる場合がある

#### 3. `sharpness_lap_var`（シャープネス）

何を測るか:
輪郭（エッジ）の明瞭さ

計算方法:
`cv2.Laplacian(gray, cv2.CV_64F).var()`

解釈:
- 高い: ピントが合っている傾向
- 低い: ピンぼけ・ブレの傾向

実測例:
`sample.jpg` に Gaussian blur をかけると `sigma_2: 3.63`、`sigma_3: 2.16` と低下し、ぼかしが強いほど値が下がる

注意:
- 平坦な被写体は値が低くなりやすい
- ノイズでエッジが増えると値が高く見えることがある

### 判定（Judgement）仕様

#### 判定ロジック

- 各指標にしきい値範囲を設定
- すべて範囲内: `OK`
- 1つでも範囲外: `NG`（外れた指標を理由として列挙）

#### 判定結果の形式

- `verdict`: `"OK"` または `"NG"`
- `reasons`: `NG` の場合の理由配列（例: `["brightness_mean=50 は許容範囲外"]`）

#### 現在のしきい値（`data/analysis/thresholds.json`, 2026-02-21時点）

| 指標 | 下限 | 上限 |
|---|---|---|
| `sharpness_lap_var` | `17.707344241118314` | なし |
| `brightness_mean` | `128.14072814941406` | `150.60439208984374` |
| `contrast_std` | `42.15484436035156` | なし |

NG 条件:
- `sharpness_lap_var < 17.707344241118314`
- `brightness_mean` が `[128.14072814941406, 150.60439208984374]` の範囲外
- `contrast_std < 42.15484436035156`

### しきい値設計（Thresholds）と根拠

#### 対象データと実行コマンド

- dataset root: `data/dataset`（Day9 で生成した `sharp` / `blur(sigma_*)`）
- 探索スクリプト: `scripts/day11_design_thresholds.py`

```bash
python scripts/day11_design_thresholds.py --dataset-root data/dataset
```

#### sharpness しきい値の決定方法

- 候補しきい値 `t` を分位点 5%〜95% の範囲で探索
- `Balanced Accuracy = (感度 + 特異度) / 2` を最大化する `t` を採用
- 探索結果（`thresholds.json`）: `meta.sharpness_search_balanced_accuracy = 0.825`

#### brightness / contrast の決め方

- sharp 画像分布から外れ値を除外するガードとして設定
- `brightness_mean`: sharp 分布の低すぎ/高すぎを弾く
- `contrast_std`: sharp 分布の低コントラスト側を弾く

## 実装構成

```text
bioimage_project/
  app/
    app.py                    # Streamlit UI
  src/bioimage_qc/
    io.py                     # 画像読み込み/グレースケール変換
    metrics.py                # 指標計算
    judge.py                  # OK/NG 判定ロジック
    pipeline.py               # 評価処理統合
    exceptions.py             # アプリ共通例外
  scripts/
    eval_and_judge.py         # 1枚評価CLI
    batch_eval.py             # 一括評価CLI + CSV出力
    day11_design_thresholds.py# しきい値設計
    self_test.py              # 最低限の自己テスト
  docs/
    bioimage_quality_check_app.md
```

## セットアップ

1. 仮想環境を作成

```bash
python3 -m venv .venv
```

2. 仮想環境を有効化

```bash
source .venv/bin/activate
```

3. 依存をインストール

```bash
pip install -r requirements.txt
```

## 実行手順

### 1. Streamlitアプリ起動

```bash
streamlit run app/app.py
```

実装済み機能:

- 画像アップロード
- プレビュー表示
- 3指標表示
- OK/NG と理由表示
- 結果CSVダウンロード（単一画像）
- 入力チェックと例外表示

### 2. 1枚画像をCLIで評価

```bash
python scripts/eval_and_judge.py data/raw/sample.jpg --sharpness-min 500
```

### 3. フォルダ一括評価（CSV出力）

```bash
python scripts/batch_eval.py data/dataset --thresholds data/analysis/thresholds.json --out outputs/results.csv
```

### 4. 自己テスト

```bash
python scripts/self_test.py
```

## 結果（現時点）

`outputs/results.csv`（25枚）を対象に確認した結果:

- 正解率: `0.76`（19/25）
- Sharp再現率: `0.60`（3/5）
- Blur再現率: `0.80`（16/20）
- Balanced Accuracy: `0.70`

注記:

- データ数が少ないため、性能値は参考値です
- 実運用時は対象データで再調整してください

## 限界

- 被写体依存: 平坦な画像は「ぼけ」と区別しづらい
- ノイズ影響: ノイズでシャープネスが過大になる場合がある
- 照明依存: 逆光/白飛びで明るさ指標が不安定になる
- 色情報未評価: 基本はグレースケール評価のため色かぶりは対象外
- しきい値固定の限界: ドメインが変わると再調整が必要

## 今後の改善

- しきい値の自動再推定（データ追加時）
- 指標追加（ノイズ量、局所コントラスト等）
- 一括処理UI（複数画像アップロード）
- テスト強化（pytest化、異常系の網羅）
- ドキュメントと実測値の同期自動化

## ライセンス/データ取り扱い

- コードのライセンスは未設定です（必要なら `LICENSE` を追加）
- `data/` 配下画像の再配布可否は画像ごとの権利条件を確認してください

## 関連ドキュメント

- `docs/bioimage_quality_check_app.md`: 1か月計画と完了条件
