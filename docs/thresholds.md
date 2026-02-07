# Thresholds（しきい値）メモ

## 1. 対象データ
- 作成日: 2026-02-07
- dataset root: data/dataset
- 生成方法: Day9（sharp / blur(sigma_*)）
- 評価コマンド:
  - python scripts/day11_design_thresholds.py --dataset-root data/dataset

## 2. 使う指標
- sharpness_lap_var: Laplacian 分散（値が小さいほどボケやすい）
- brightness_mean: 明るさ平均（暗すぎ/白飛びの検知）
- contrast_std: コントラスト（低コントラストの検知）

## 3. 採用したしきい値（初期値）
- sharpness_lap_var_min: 2.16
- brightness_mean_min: 137.19
- brightness_mean_max: 158.55
- contrast_std_min: 11.39
(thresholds.jsonより)

判定ルール（初期案）:
- NG if
  - sharpness_lap_var < sharpness_lap_var_min
  - OR brightness_mean が [min, max] の範囲外
  - OR contrast_std < contrast_std_min

## 4. しきい値の決め方（根拠）
- sharpness の境界値 t を候補集合（分位点 5%〜95% の間を細かく）から探索し、
  Balanced Accuracy が最大になる t を採用した。
- Balanced Accuracy = (sharp再現率 + blur再現率)/2
  - sharp再現率: sharp を見逃さない力
  - blur再現率: blur を取りこぼさない力

### 4.1 sharpness（主判定）
- データの傾向:
  - Sharp画像（5枚）: sharpness = 2.19〜271.30（平均87.9）
  - Blur画像（20枚）: sharpness = 0.77〜271.30
    - sigma=0: 2.05〜271.30（ぼかし無しのため高め）
    - sigma≥1: ほぼ全て < 17
- 採用した境界値: **2.16**
  - この値で、Balanced Accuracy = 0.80 を達成

### 4.2 brightness / contrast（ガード）
- sharp 画像の分布から、極端な外れを弾く目的で設定
  - brightness_min/max: sharp の 1%点〜99%点
  - contrast_min: sharp の 1%点

## 5. 現時点の評価
- Balanced Accuracy（探索時の最大値）: 0.80 (80%)
※ これはsharpness指標のみで評価した場合の値
- 混同行列（sharp=OK, blur=NG）:
  - TP(OKをOK): 2
  - TN(NGをNG): 15
  - FP(NGをOKに誤判定): 5
  - FN(OKをNGに誤判定): 3
- 実際の性能（全しきい値適用後）:
  - 正解率: 68% (17/25)
  - Sharp再現率: 40% (2/5) 
  - Blur再現率: 75% (15/20)

## 6. 今後の調整方針
- 「ボケ見逃しを減らす」→ sharpness_min を上げる（NGが増える）
- 「OK落としを減らす」→ sharpness_min を下げる（見逃しが増える可能性）
- UI（Week3）でスライダー調整できる前提で、ここは初期値として運用する
