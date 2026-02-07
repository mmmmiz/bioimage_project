# Bioimage Quality Check App  
開発タスク計画（1か月・80時間）

## 前提条件

- 開発期間：4週間（約1か月）
- 作業時間：
  - 平日：2時間／日
  - 土日：5時間／日
- 合計作業時間：約80時間
- 実行環境：ローカル（Mac）
- 開発言語：Python
- UI：Streamlit
- 使用技術：NumPy / OpenCV / pandas / matplotlib（検証用）/ jupyter notebook(VScode上)　
- 深層学習（PyTorch / TensorFlow）は使用しない
### 補足（任意）：Notebook運用
- Notebookで試行錯誤を推奨。ただし scripts/ に同等の確認コードを残す
- VS Code の Notebook UI（.ipynb）を検証用に使用してよい
- ただし成果物（提出・再現性の正本）は scripts/ 配下の .py とする
- Notebookは相対パスの基準（cwd）がズレやすいので、BASE_DIR（プロジェクトルート）を決めてから data/ を参照する

---

## Week 1：基礎固め（画像を「扱える」状態にする）【20h】

### Day 1（平日・2h）
**目的**：開発環境構築と画像読み込み確認  
- Python仮想環境（venv）作成  
- ライブラリ導入（numpy, opencv-python, matplotlib, pandas, streamlit）  
- OpenCVで画像を読み込み、shape / dtypeを確認  

**成果物**
- requirements.txt
- 画像を読み込める確認コード

**基礎確認**
- Python仮想環境
- import文
- NumPy配列の shape / dtype

---

### Day 2（平日・2h）
**目的**：画像＝配列の理解  
- カラー画像とグレースケール画像の違い確認  
- グレースケール変換  
- matplotlibによる画像表示 

**成果物**
- 画像表示スクリプト

**基礎確認**
- NumPy配列
- matplotlibの基本

---

### Day 3（平日・2h）
**目的**：指標① 明るさ（Brightness）実装  
- 平均輝度の計算（np.mean）  
- 関数化  

**成果物**
- calc_brightness() 関数

**基礎確認**
- Python関数
- 平均（mean）

---

### Day 4（平日・2h）
**目的**：指標② コントラスト（Contrast）実装  
- 標準偏差の計算（np.std）  

**成果物**
- calc_contrast() 関数

**基礎確認**
- 標準偏差（std）

---

### Day 5（平日・2h）
**目的**：指標③ シャープネス（ピンぼけ）実装  
- Laplacianフィルタ  
- 分散計算（np.var）  

**成果物**
- calc_sharpness() 関数（Laplacian分散）

**基礎確認**
- フィルタ処理
- 分散（variance）

---

### Day 6（土日・5h）
**目的**：指標計算を統合  
- 画像読み込み → 指標計算までの一連処理  
- CLIで1枚画像を評価  

**成果物**
- 指標計算フロー完成

---

### Day 7（土日・5h）
**目的**：安定化・リファクタリング  
- カラー/グレー両対応  
- エラーハンドリング  
- コード整理  

---

## Week 2：人工ピンぼけ生成＋判定設計【20h】

### Day 8（平日・2h）
**目的**：人工ピンぼけ生成  
- GaussianBlurの実装  
- σ（ぼけ強度）指定  

**成果物**
- synth_blur.py

---

### Day 9（平日・2h）
**目的**：データセット作成  
- σ = 0,1,2,3 などで画像生成  
- フォルダ整理（sharp / blur）  

**現在のディレクトリ構成（2026-02-07時点）**
```text
bioimage_project/
  README.md
  requirements.txt
  data/
    raw/
    dataset/
      sharp/
      blur/
        sigma_0/
        sigma_1/
        sigma_2/
        sigma_3/
  docs/
    bioimage_quality_check_app.md
    lecture/
      html/
      md/
  notebooks/
    day02.ipynb
    day04.ipynb
    day05.ipynb
  scripts/
    check_image.py
    check_sharpness.py
    day02_view_image.py
    day03_brightness_demo.py
    eval_image.py
    make_dataset.py
    metrics_demo.py
    self_test.py
    synth_blur.py
  src/
    bioimage_qc/
      __init__.py
      exceptions.py
      io.py
      metrics.py
      pipeline.py
```

---

### Day 10（平日・2h）
**目的**：指標挙動確認  
- ぼけ強度と指標値の関係確認  
- matplotlibで可視化  

---

### Day 11（平日・2h）
**目的**：しきい値設計  
- OK / NG基準値決定  

---

### Day 12（平日・2h）
**目的**：判定ロジック実装  
- OK / NG判定  
- NG理由付与  

**成果物**
- judge.py

---

### Day 13（土日・5h）
**目的**：処理統合  
- 画像 → 指標 → 判定 → 結果dict化  

---

### Day 14（土日・5h）
**目的**：ドキュメント下書き  
- 指標・判定方法の説明作成  

---

## Week 3：Streamlit UI＋CSV出力【20h】

### Day 15（平日・2h）
**目的**：Streamlit導入  
- アプリ起動  
- 画像アップロード  

---

### Day 16（平日・2h）
**目的**：指標表示  
- 数値を表形式で表示  

---

### Day 17（平日・2h）
**目的**：しきい値UI化  
- スライダー実装  

---

### Day 18（平日・2h）
**目的**：判定結果表示  
- OK / NG と理由表示  

---

### Day 19（平日・2h）
**目的**：CSV出力  
- pandasでDataFrame生成  
- ダウンロードボタン  

---

### Day 20（土日・5h）
**目的**：UI安定化  
- 入力チェック  
- 例外処理  

---

### Day 21（土日・5h）
**目的**：画面最終調整  
- 説明文追加  
- 見やすさ改善  

---

## Week 4：仕上げ・応募準備【20h】

### Day 22（平日・2h）
**目的**：ディレクトリ整理  
- src分割  
- import整理  

---

### Day 23（平日・2h）
**目的**：再現性確保  
- requirements.txt最終化  
- 実行手順整理  

---

### Day 24（平日・2h）
**目的**：サンプル画像整理  
- ライセンス確認  

---

### Day 25（平日・2h）
**目的**：README完成  
- 背景 / 方法 / 結果 / 限界 / 今後  

---

### Day 26（平日・2h）
**目的**：応募書類用整理  
- 実装内容の文章化  

---

### Day 27（土日・5h）
**目的**：余力対応  
- 一括処理（任意）  
- 改善  

---

### Day 28（土日・5h）
**目的**：最終確認  
- クリーン環境想定テスト  
- スクリーンショット作成  

---

## 完了条件

- Streamlitアプリがローカルで起動する
- 画像品質評価が実行できる
- CSVダウンロードが可能
- READMEで設計・実装を説明できる
- 応募書類に記載できる成果物になっている
