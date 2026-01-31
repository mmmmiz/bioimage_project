# Day 4: Streamlit基礎（3〜4時間）

---

## 📋 今日のゴール

Pythonだけでウェブアプリを作れる「Streamlit」の基本を習得します。  
HTMLやCSSの知識がなくても、画像をアップロードしたり、スライダーで値を調整したりできるアプリが作れるようになります。

---

## 🎯 学習目標

| # | 目標 | 確認 |
|---|------|------|
| 1 | Streamlitの基本概念を理解し、アプリを起動できる | ☐ |
| 2 | `st.file_uploader()`で画像をアップロードできる | ☐ |
| 3 | `st.slider()`でしきい値を調整できる | ☐ |
| 4 | `st.dataframe()`でデータを表形式で表示できる | ☐ |
| 5 | `st.download_button()`でCSVをダウンロードできる | ☐ |

---

## 📚 用語表

| 用語 | 読み方 | 意味（かんたんに） | 意味（くわしく） |
|------|--------|-------------------|-----------------|
| Streamlit | ストリームリット | Pythonでウェブアプリを作るツール | Pythonスクリプトを実行するだけでインタラクティブなウェブアプリケーションを生成できるフレームワーク |
| ウィジェット | ウィジェット | 画面上の部品（ボタンやスライダー） | ユーザーが操作できるUI要素。入力フォーム、ボタン、スライダーなどの総称 |
| レンダリング | レンダリング | 画面に表示すること | データやコードを視覚的な形式に変換して画面に描画すること |
| 再実行 | さいじっこう | スクリプトを最初から実行し直すこと | Streamlitの特徴で、ユーザーが何か操作するたびにスクリプト全体が上から順に実行される |
| session_state | セッションステート | データを保存しておく場所 | ページの再実行をまたいでデータを保持するためのStreamlitの仕組み |
| DataFrame | データフレーム | 表形式のデータ | pandasライブラリで使う2次元のデータ構造（行と列を持つ表） |

---

## 4.1 Streamlitとは何か

### ざっくり説明（まずはここから！）

Streamlitは「**Pythonコードを書くだけでウェブアプリが作れる**」魔法のようなツールです。

通常、ウェブアプリを作るには：
- HTML（画面の構造）
- CSS（見た目のデザイン）
- JavaScript（動きをつける）

これらの知識が必要ですが、Streamlitなら**Pythonだけ**で全部できます！

```
【従来のウェブアプリ開発】
Python（バックエンド） + HTML + CSS + JavaScript（フロントエンド）
         ↓ めんどう...

【Streamlitの世界】
Python だけ！
         ↓ かんたん！
```

### くわしい説明

Streamlitには3つの大きな特徴があります：

#### 特徴1: 上から順に実行される

```python
# このコードは上から順番に実行され、
# 書いた順番に画面に表示される
st.title("タイトル")      # ← 1番目に表示
st.write("本文")          # ← 2番目に表示
st.button("ボタン")       # ← 3番目に表示
```

#### 特徴2: 操作があると全部再実行される

ボタンを押したり、スライダーを動かしたりすると、**スクリプト全体が最初から実行し直されます**。

```
【ユーザーがスライダーを動かす】
        ↓
【スクリプト全体が再実行される】
        ↓
【新しい値で画面が更新される】
```

これは最初は不思議に感じますが、慣れるととても直感的です。

#### 特徴3: 状態の保持にはsession_stateを使う

再実行されると変数は初期化されてしまうため、データを保持したい場合は`st.session_state`を使います（後で詳しく学びます）。

### 最小限のアプリを作ってみよう

```python
# app.py
import streamlit as st

st.title("Hello Streamlit!")
st.write("これは最初のStreamlitアプリです")
```

### 起動方法

ターミナルで以下のコマンドを実行します：

```bash
streamlit run app.py
```

すると、ブラウザが自動的に開いて、アプリが表示されます！

> 💡 **ポイント**: `python app.py`ではなく`streamlit run app.py`で起動します

### 🔍 ここがわからなければ確認しよう

- **ターミナルの使い方がわからない**: macOSの「ターミナル」アプリの基本操作を確認
- **`import`がわからない**: Pythonのモジュールとインポートの基礎を確認
- **ファイルの保存場所がわからない**: カレントディレクトリとパスの概念を確認

### 🔗 公式リファレンス

- [Streamlit公式ドキュメント](https://docs.streamlit.io/)
- [Get started](https://docs.streamlit.io/get-started)

---

## 4.2 ファイルアップロード

### ざっくり説明

`st.file_uploader()`を使うと、ユーザーが画像やファイルをアップロードできる機能が作れます。

```
【ユーザーの操作】
画像ファイルを選択 → アップロード → Pythonで処理

【コードで書くと】
uploaded_file = st.file_uploader("画像を選択")
```

### くわしい説明

#### 基本的な使い方

```python
import streamlit as st

# ファイルアップロードのウィジェットを表示
uploaded_file = st.file_uploader(
    "画像を選択してください",  # 表示するラベル
    type=["jpg", "jpeg", "png"]  # 許可する拡張子
)
```

**重要ポイント**:
- `uploaded_file`は最初は`None`（何もない状態）
- ユーザーがファイルを選択すると、値が入る
- `type`で許可するファイルの種類を制限できる

#### OpenCVで画像を処理する

アップロードされたファイルをOpenCVで扱うには、少し変換が必要です：

```python
import streamlit as st
import cv2
import numpy as np

st.title("画像アップローダー")

# ファイルアップロード
uploaded_file = st.file_uploader(
    "画像を選択してください",
    type=["jpg", "jpeg", "png"]
)

# ファイルがアップロードされたかチェック
if uploaded_file is not None:
    # ステップ1: ファイルの中身をバイト列（数字の羅列）として読み込む
    file_bytes = np.asarray(
        bytearray(uploaded_file.read()),  # ファイルを読み込んでバイト配列に
        dtype=np.uint8                     # 0〜255の整数として扱う
    )
    
    # ステップ2: バイト列を画像としてデコード（解読）する
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    # ステップ3: 画像を表示
    st.image(image, channels="BGR", caption="アップロードされた画像")
    
    # 画像の情報を表示
    st.write(f"サイズ: {image.shape}")
```

#### コードの流れを図解

```
【ユーザーが画像をアップロード】
        ↓
uploaded_file（ファイルオブジェクト）
        ↓
uploaded_file.read()（バイトデータを読み込む）
        ↓
bytearray()（バイト配列に変換）
        ↓
np.asarray()（NumPy配列に変換）
        ↓
cv2.imdecode()（画像としてデコード）
        ↓
image（OpenCVで扱えるNumPy配列！）
```

#### `st.image()`の注意点

OpenCVは**BGR形式**（青・緑・赤の順）で画像を扱いますが、`st.image()`はデフォルトで**RGB形式**を期待します。

```python
# 方法1: channels="BGR"を指定する（推奨）
st.image(image, channels="BGR")

# 方法2: RGBに変換してから表示する
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
st.image(image_rgb)
```

### 💡 プロジェクトでの使用場面

- **Day 15**: 画像アップロード機能の実装

### 🔍 ここがわからなければ確認しよう

- **`if uploaded_file is not None:`がわからない**: Pythonの条件分岐とNone判定を確認
- **バイト・バイト配列がわからない**: Pythonのバイナリデータの基礎を確認
- **`cv2.imdecode()`がわからない**: Day 2のOpenCV基礎を復習

### 🔗 公式リファレンス

- [st.file_uploader](https://docs.streamlit.io/develop/api-reference/widgets/st.file_uploader)

---

## 4.3 スライダー

### ざっくり説明

`st.slider()`を使うと、ユーザーがマウスでドラッグして数値を調整できるスライダーが作れます。

```
【画面のイメージ】
明るさのしきい値
[====●========] 100

【コードで書くと】
value = st.slider("明るさのしきい値", 0, 255, 100)
```

### くわしい説明

#### 基本的なスライダー

```python
import streamlit as st

st.title("しきい値調整")

# 基本のスライダー
brightness_threshold = st.slider(
    "明るさのしきい値",  # ラベル
    min_value=0,         # 最小値
    max_value=255,       # 最大値
    value=100,           # デフォルト値（初期値）
    step=1               # 刻み幅（1ずつ変化）
)

st.write(f"現在の値: {brightness_threshold}")
```

**パラメータの意味**:

| パラメータ | 説明 | 例 |
|-----------|------|-----|
| 第1引数 | スライダーのラベル（説明文） | `"明るさのしきい値"` |
| `min_value` | 選べる最小の値 | `0` |
| `max_value` | 選べる最大の値 | `255` |
| `value` | 最初に選ばれている値 | `100` |
| `step` | 値の変化量（刻み幅） | `1` |

#### 範囲スライダー（2つの値を選ぶ）

`value`にタプル（2つの値のペア）を渡すと、**範囲を選択するスライダー**になります。

```python
import streamlit as st

# 範囲スライダー
contrast_range = st.slider(
    "コントラストの範囲",
    min_value=0.0,
    max_value=100.0,
    value=(20.0, 80.0),  # ← タプルで範囲を指定！
    step=0.5
)

# 結果はタプルで返ってくる
lower = contrast_range[0]  # 下限値
upper = contrast_range[1]  # 上限値

st.write(f"下限: {lower}, 上限: {upper}")
```

```
【画面のイメージ】
コントラストの範囲
[====●==========●====]
    20.0         80.0
```

#### 整数と小数の違い

```python
# 整数のスライダー（min_value, max_valueがint）
int_slider = st.slider("整数", 0, 100, 50)
# → 戻り値もint型

# 小数のスライダー（min_value, max_valueがfloat）
float_slider = st.slider("小数", 0.0, 100.0, 50.0)
# → 戻り値もfloat型
```

> 💡 **ポイント**: 小数を扱いたい場合は、必ず`0.0`のように小数点をつけて書きましょう

### 💡 プロジェクトでの使用場面

- **Day 17**: しきい値のUI化（ユーザーが判定基準を調整できるようにする）

### 🔍 ここがわからなければ確認しよう

- **タプルがわからない**: Pythonのタプル`(a, b)`の基礎を確認
- **`int`と`float`の違いがわからない**: Pythonのデータ型の基礎を確認
- **インデックス`[0]`や`[1]`がわからない**: Pythonのシーケンスとインデックスを確認

### 🔗 公式リファレンス

- [st.slider](https://docs.streamlit.io/develop/api-reference/widgets/st.slider)

---

## 4.4 テーブル表示

### ざっくり説明

`st.dataframe()`や`st.table()`を使うと、データを**表形式**で画面に表示できます。

```
【表示イメージ】
┌──────────┬────────┬────────┐
│ 指標     │ 値     │ 判定   │
├──────────┼────────┼────────┤
│ 明るさ   │ 125.5  │ OK     │
│ コントラスト │ 45.2 │ OK   │
└──────────┴────────┴────────┘
```

### くわしい説明

#### `st.dataframe()` vs `st.table()` の違い

| 機能 | `st.dataframe()` | `st.table()` |
|------|------------------|--------------|
| ソート（並び替え） | ✅ できる | ❌ できない |
| スクロール | ✅ できる | ❌ できない |
| 行のハイライト | ✅ できる | ❌ できない |
| 用途 | 大きなデータ、インタラクティブ | 小さなデータ、固定表示 |

#### 基本的な使い方

```python
import streamlit as st
import pandas as pd

# サンプルデータを作成
data = {
    "指標": ["明るさ", "コントラスト", "シャープネス"],
    "値": [125.5, 45.2, 1250.0],
    "判定": ["OK", "OK", "NG"]
}
df = pd.DataFrame(data)

# インタラクティブなテーブル（おすすめ）
st.subheader("st.dataframe（インタラクティブ）")
st.dataframe(df, use_container_width=True)

# 静的なテーブル
st.subheader("st.table（静的）")
st.table(df)
```

#### `use_container_width=True`とは

テーブルの幅を画面いっぱいに広げるオプションです。

```python
# 幅が狭い（デフォルト）
st.dataframe(df)

# 幅を画面いっぱいに広げる
st.dataframe(df, use_container_width=True)
```

#### 辞書のリストからDataFrameを作る

```python
import pandas as pd

# 辞書のリスト形式（よく使うパターン）
results = [
    {"filename": "image1.jpg", "brightness": 125.5, "status": "OK"},
    {"filename": "image2.jpg", "brightness": 98.3, "status": "NG"},
    {"filename": "image3.jpg", "brightness": 150.2, "status": "OK"},
]

# DataFrameに変換
df = pd.DataFrame(results)

# 表示
st.dataframe(df)
```

このパターンは、画像を1枚ずつ処理して結果をリストに追加していく場合に便利です：

```python
results = []  # 空のリスト

# 画像ごとに処理（イメージ）
for image_file in image_files:
    # ... 画像を処理 ...
    results.append({
        "filename": image_file.name,
        "brightness": calculated_brightness,
        "status": "OK" if calculated_brightness > threshold else "NG"
    })

# 最後にDataFrameに変換して表示
df = pd.DataFrame(results)
st.dataframe(df)
```

### 💡 プロジェクトでの使用場面

- **Day 16**: 指標の表形式表示

### 🔍 ここがわからなければ確認しよう

- **辞書`{}`がわからない**: Pythonの辞書型の基礎を確認
- **リスト`[]`がわからない**: Pythonのリスト型の基礎を確認
- **pandasがわからない**: Day 5で詳しく学びます！

### 🔗 公式リファレンス

- [st.dataframe](https://docs.streamlit.io/develop/api-reference/data/st.dataframe)
- [st.table](https://docs.streamlit.io/develop/api-reference/data/st.table)

---

## 4.5 ダウンロードボタン

### ざっくり説明

`st.download_button()`を使うと、処理結果をCSVファイルなどでダウンロードできるボタンが作れます。

```
【画面のイメージ】
[📥 CSVをダウンロード]  ← クリックするとファイルがダウンロードされる
```

### くわしい説明

#### 基本的な使い方

```python
import streamlit as st
import pandas as pd

# データ作成
df = pd.DataFrame({
    "ファイル名": ["image1.jpg", "image2.jpg"],
    "明るさ": [125.5, 98.3],
    "判定": ["OK", "NG"]
})

# DataFrameをCSV形式の文字列に変換
csv = df.to_csv(index=False)

# ダウンロードボタンを表示
st.download_button(
    label="📥 CSVをダウンロード",   # ボタンに表示するテキスト
    data=csv,                       # ダウンロードするデータ
    file_name="quality_check_result.csv",  # 保存時のファイル名
    mime="text/csv"                 # ファイルの種類（MIMEタイプ）
)
```

#### パラメータの意味

| パラメータ | 説明 | 例 |
|-----------|------|-----|
| `label` | ボタンに表示するテキスト | `"📥 CSVをダウンロード"` |
| `data` | ダウンロードするデータ（文字列やバイト列） | `csv` |
| `file_name` | ダウンロード時のファイル名 | `"result.csv"` |
| `mime` | データの種類を示す文字列 | `"text/csv"` |

#### MIMEタイプとは

**MIME（マイム）タイプ**は、ファイルの種類をコンピュータに伝えるための文字列です。

| ファイルの種類 | MIMEタイプ |
|---------------|-----------|
| CSV | `text/csv` |
| JSON | `application/json` |
| 画像（PNG） | `image/png` |
| 画像（JPEG） | `image/jpeg` |

#### `df.to_csv(index=False)`とは

DataFrameをCSV形式の文字列に変換します。

```python
import pandas as pd

df = pd.DataFrame({
    "名前": ["田中", "鈴木"],
    "年齢": [25, 30]
})

# index=True（デフォルト）だと行番号も出力される
print(df.to_csv())
# 出力:
# ,名前,年齢
# 0,田中,25
# 1,鈴木,30

# index=False だと行番号は出力されない
print(df.to_csv(index=False))
# 出力:
# 名前,年齢
# 田中,25
# 鈴木,30
```

> 💡 **ポイント**: 通常は`index=False`を使うことが多いです（行番号は不要なことが多いため）

### 💡 プロジェクトでの使用場面

- **Day 19**: CSV出力機能の実装

### 🔍 ここがわからなければ確認しよう

- **`df.to_csv()`がわからない**: Day 5のpandas基礎で詳しく学びます
- **絵文字の入力方法がわからない**: macOSでは`Control + Command + Space`で絵文字パレットが開きます

### 🔗 公式リファレンス

- [st.download_button](https://docs.streamlit.io/develop/api-reference/widgets/st.download_button)

---

## 4.6 まとめ：Streamlitの基本パターン

### よく使うパターンの組み合わせ

```python
import streamlit as st
import cv2
import numpy as np
import pandas as pd

# ===== タイトル =====
st.title("画像品質チェッカー")

# ===== ファイルアップロード =====
uploaded_file = st.file_uploader("画像を選択", type=["jpg", "png"])

# ===== しきい値調整 =====
brightness_threshold = st.slider("明るさのしきい値", 0, 255, 100)

# ===== メイン処理 =====
if uploaded_file is not None:
    # 画像を読み込み
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    # 画像を表示
    st.image(image, channels="BGR", caption="アップロードされた画像")
    
    # 明るさを計算
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    brightness = float(np.mean(gray))
    
    # 判定
    status = "OK" if brightness >= brightness_threshold else "NG"
    
    # 結果をDataFrameで表示
    result = pd.DataFrame([{
        "ファイル名": uploaded_file.name,
        "明るさ": round(brightness, 2),
        "判定": status
    }])
    st.dataframe(result, use_container_width=True)
    
    # CSVダウンロード
    csv = result.to_csv(index=False)
    st.download_button("📥 CSVをダウンロード", csv, "result.csv", "text/csv")
```

### 処理の流れ

```
1. タイトルを表示
        ↓
2. ファイルアップロード部品を表示
        ↓
3. スライダーを表示
        ↓
4. ファイルがアップロードされたら...
        ↓
5. 画像を読み込んで処理
        ↓
6. 結果を表に表示
        ↓
7. ダウンロードボタンを表示
```

---

## ✅ Day 4 確認問題

以下の問題に答えてみてください。

### 問題1: ファイルアップロード

以下のコードで、ファイルがアップロードされていない場合、if文の中は実行される？

```python
uploaded_file = st.file_uploader("ファイルを選択")
if uploaded_file:
    st.write("ファイルがあります")
```

### 問題2: スライダー

`st.slider()`で範囲（下限と上限）を選択するにはどうする？

- A: `value=(10, 90)` とタプルで指定
- B: `range=True` と指定
- C: 範囲選択はできない

### 問題3: テーブル表示

`st.dataframe()`と`st.table()`の違いは何？

### 問題4: 再実行の仕組み

Streamlitでスライダーを動かすと何が起こる？

<details>
<summary>解答を見る</summary>

### 問題1の解答

**実行されない**

`uploaded_file`は最初`None`（値がない状態）です。  
Pythonでは`None`は`False`として扱われる（Falsyな値）ため、if文の条件が`False`となり、中のコードは実行されません。

```python
# uploaded_file が None の場合
if None:  # → False として扱われる
    # ここは実行されない
```

---

### 問題2の解答

**A: `value=(10, 90)` とタプルで指定**

```python
# 範囲スライダーの例
range_value = st.slider(
    "範囲を選択",
    min_value=0,
    max_value=100,
    value=(10, 90)  # ← タプルで指定すると範囲スライダーになる
)
```

---

### 問題3の解答

| 機能 | `st.dataframe()` | `st.table()` |
|------|------------------|--------------|
| ソート（並び替え） | ✅ できる | ❌ できない |
| スクロール | ✅ できる | ❌ できない |
| 操作性 | インタラクティブ | 静的 |

- `st.dataframe()`: 列をクリックしてソートしたり、大きなデータをスクロールしたりできる
- `st.table()`: 表示するだけで操作はできない

---

### 問題4の解答

**スクリプト全体が最初から再実行される**

Streamlitでは、ユーザーが何か操作（スライダーを動かす、ボタンを押す等）をするたびに、Pythonスクリプトが**上から下まで全部**実行し直されます。

これにより：
- 新しいスライダーの値で計算がやり直される
- 画面が新しい結果で更新される

</details>

---

## 📝 Day 4 完了チェックリスト

以下の項目をすべて確認できたら、Day 4は完了です！

- [ ] `streamlit run app.py`でアプリを起動できる
- [ ] `st.file_uploader()`でファイルをアップロードできる
- [ ] アップロードした画像をOpenCVで処理できる
- [ ] `st.slider()`で数値を調整できる
- [ ] 範囲スライダー（タプル指定）を使える
- [ ] `st.dataframe()`でデータを表形式で表示できる
- [ ] `st.download_button()`でCSVをダウンロードできる
- [ ] Streamlitの「再実行」の仕組みを理解している

---

## 🎉 お疲れさまでした！

Day 4では、Streamlitの基本的な部品（ウィジェット）の使い方を学びました。

これで、画像をアップロードして、しきい値を調整して、結果を表示して、CSVでダウンロードするという、アプリの基本的な流れが作れるようになりました！

次のDay 5では、pandasの基礎と、これまで学んだ内容の総復習を行います。
