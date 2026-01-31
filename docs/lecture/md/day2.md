# Day 2: OpenCV基礎（2〜3時間）

---

## 📋 この日の学習について

**OpenCV（オープンシーブイ）** は、画像処理や動画処理を行うためのライブラリです。  
この日は、画像を読み込んで処理するための基本的な操作を学びます。

### この日のゴール
- 画像をプログラムで読み込めるようになる
- カラー画像とグレースケール画像の違いを理解する
- シャープネス（ピントの合い具合）を測る方法を知る
- 画像をぼかす方法を知る

### 前提知識
- Day 1のNumPy基礎（配列、`shape`、`dtype`など）
- Pythonの基本文法（import、変数、関数呼び出し）

---

## 2.1 画像の読み込みと表示

### 学習目標
- [ ] `cv2.imread()`で画像を読み込める
- [ ] OpenCVがBGR形式で読み込むことを理解する

---

### 🔰 ざっくり説明（まずはここから）

**画像を読み込む**というのは、画像ファイル（JPGやPNGなど）をプログラムで扱える形（数字の集まり）に変換することです。

OpenCVでは `cv2.imread()` という命令を使います。  
読み込まれた画像は、Day 1で学んだ**NumPy配列**として扱われます。

```
画像ファイル → cv2.imread() → NumPy配列（数字の集まり）
```

---

### 📚 詳しい説明

#### 2.1.1 画像の読み込み

```python
import cv2  # OpenCVライブラリを読み込む

# 画像を読み込む
image = cv2.imread('sample.jpg')

# 読み込めたか確認（とても重要！）
if image is None:
    print("画像が読み込めませんでした")
else:
    print(f"画像サイズ: {image.shape}")
```

**コードの解説**:

| コード | 意味 |
|--------|------|
| `import cv2` | OpenCVライブラリを使えるようにする |
| `cv2.imread('sample.jpg')` | 画像ファイルを読み込む |
| `image is None` | 読み込みに失敗したかチェック |
| `image.shape` | 画像のサイズ（高さ, 幅, チャンネル数）を取得 |

---

#### ⚠️ 重要ポイント：読み込み失敗の検知

**`cv2.imread()`の注意点**:
- 読み込みに失敗しても**エラーが出ない**（プログラムが止まらない）
- 失敗すると**`None`**という特別な値が返ってくる
- 必ず`if image is None:`でチェックする習慣をつけよう

```python
# 存在しないファイルを読み込もうとした場合
image = cv2.imread('sonzai_shinai.jpg')
print(image)  # None と表示される（エラーにならない！）
```

> 💡 **なぜエラーにならない？**  
> OpenCVはC++で作られたライブラリで、Pythonとは異なる設計思想を持っています。
> エラーを投げる代わりに`None`を返すことで、プログラマーが自分で対処できるようになっています。

---

#### 2.1.2 BGR形式について

**BGR（ビージーアール）** とは、色の並び順のことです。

| 形式 | 色の順番 | 使う場面 |
|------|----------|----------|
| **BGR** | 青(Blue) → 緑(Green) → 赤(Red) | OpenCV |
| **RGB** | 赤(Red) → 緑(Green) → 青(Blue) | 一般的な画像形式、matplotlib |

```python
import cv2

image = cv2.imread('sample.jpg')

# 最初のピクセルの色を確認
# 順番は [青, 緑, 赤] になっている
print(image[0, 0])  # 例: [255, 128, 64] → 青が255, 緑が128, 赤が64
```

> 💡 **なぜBGRなの？**  
> 歴史的な理由です。OpenCVが生まれた1990年代後半、WindowsのBMP形式がBGR順だったため、
> OpenCVもそれに合わせました。今でも互換性のためにBGRを使い続けています。

---

#### 2.1.3 グレースケール読み込み

**グレースケール（白黒画像）** で読み込むこともできます。

```python
# グレースケールで直接読み込む
gray = cv2.imread('sample.jpg', cv2.IMREAD_GRAYSCALE)

print(gray.shape)  # (高さ, 幅) ← 2次元になる（チャンネルがない）
```

**カラー画像とグレースケール画像の違い**:

| 種類 | shapeの例 | 次元数 | 説明 |
|------|-----------|--------|------|
| カラー | `(480, 640, 3)` | 3次元 | 高さ×幅×3色(BGR) |
| グレースケール | `(480, 640)` | 2次元 | 高さ×幅のみ |

---

### 💡 プロジェクトでの使用場面

- **Day 1**: 画像を読み込んで`shape`と`dtype`を確認
- **全体を通じて**: 画像ファイルの読み込みに使用

---

### 📖 基礎事項の確認ポイント

もしこのセクションがわからなければ、以下を確認しましょう：

| わからないこと | 確認する内容 |
|----------------|--------------|
| `import`文 | Pythonのモジュール読み込み |
| `if`文 | 条件分岐の基本 |
| `None` | Pythonの特殊な値（「何もない」を表す） |
| `.shape` | Day 1のNumPy配列の属性 |

---

### 🔗 公式リファレンス

- [cv2.imread - OpenCV公式ドキュメント](https://docs.opencv.org/4.x/d4/da8/group__imgcodecs.html#ga288b8b3da0892bd651fce07b3bbd3a56)
- [cv2.IMREAD_GRAYSCALE などのフラグ](https://docs.opencv.org/4.x/d8/d6a/group__imgcodecs__flags.html)

---

## 2.2 色空間の変換

### 学習目標
- [ ] `cv2.cvtColor()`でカラー変換ができる
- [ ] BGR → グレースケール変換を理解する

---

### 🔰 ざっくり説明（まずはここから）

**色空間の変換**とは、画像の色の表現方法を変えることです。

例えば：
- カラー写真を白黒写真に変える
- OpenCVのBGR形式をRGB形式に変える

これには `cv2.cvtColor()` という命令を使います。

```
カラー画像 → cv2.cvtColor() → グレースケール画像
```

---

### 📚 詳しい説明

#### 2.2.1 グレースケール変換

```python
import cv2

# カラー画像を読み込み
color_image = cv2.imread('sample.jpg')

# グレースケールに変換
gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

# 変換前後のサイズを確認
print(f"カラー: {color_image.shape}")  # (480, 640, 3) ← 3チャンネル
print(f"グレー: {gray_image.shape}")   # (480, 640)    ← 1チャンネル（2次元）
```

**コードの解説**:

| コード | 意味 |
|--------|------|
| `cv2.cvtColor(画像, 変換コード)` | 色空間を変換する |
| `cv2.COLOR_BGR2GRAY` | BGR形式 → グレースケールに変換 |

---

#### 🧮 グレースケール変換の仕組み

グレースケール変換は、3つの色(BGR)を1つの明るさの値に変換します。

**計算式**:
```
グレー値 = 0.114 × 青 + 0.587 × 緑 + 0.299 × 赤
```

> 💡 **なぜ均等に足さないの？**  
> 人間の目は緑色に最も敏感で、青色には鈍感です。
> この計算式は人間の目の感度に合わせて作られています。
> （専門用語：**輝度（Luminance）**の計算式）

---

#### ❓ なぜグレースケール変換が必要？

| 理由 | 説明 |
|------|------|
| **計算が簡単になる** | 3チャンネル → 1チャンネルでデータ量が1/3に |
| **指標計算に必要** | 明るさやシャープネスはグレースケールで計算する |
| **処理速度が上がる** | データ量が減るので計算が速い |

---

#### 2.2.2 BGR → RGB変換（matplotlib表示用）

matplotlibで画像を表示するときは、RGB形式に変換が必要です。

```python
import cv2
import matplotlib.pyplot as plt

# OpenCVで読み込み（BGR形式）
image = cv2.imread('sample.jpg')

# RGB形式に変換
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# matplotlibで表示
plt.imshow(image_rgb)
plt.title("サンプル画像")
plt.axis('off')  # 軸を非表示
plt.show()
```

**変換しないとどうなる？**
```python
# 変換せずに表示すると...
plt.imshow(image)  # 赤と青が入れ替わった変な色になる！
```

---

#### 📋 よく使う変換コード一覧

| 変換コード | 意味 |
|------------|------|
| `cv2.COLOR_BGR2GRAY` | BGR → グレースケール |
| `cv2.COLOR_BGR2RGB` | BGR → RGB |
| `cv2.COLOR_RGB2BGR` | RGB → BGR |
| `cv2.COLOR_BGR2HSV` | BGR → HSV（色相・彩度・明度） |

---

### 💡 プロジェクトでの使用場面

- **Day 2**: カラー/グレースケールの違い確認
- **Day 3〜5**: 指標計算の前処理（グレースケールに変換してから計算）

---

### 📖 基礎事項の確認ポイント

| わからないこと | 確認する内容 |
|----------------|--------------|
| 関数の引数 | Pythonの関数呼び出し（引数の渡し方） |
| 定数（`cv2.COLOR_BGR2GRAY`など） | Pythonの定数・列挙型 |

---

### 🔗 公式リファレンス

- [cv2.cvtColor - OpenCV公式ドキュメント](https://docs.opencv.org/4.x/d8/d01/group__imgproc__color__conversions.html#ga397ae87e1288a81d2363b61574eb8cab)
- [色変換コード一覧](https://docs.opencv.org/4.x/d8/d01/group__imgproc__color__conversions.html)

---

## 2.3 Laplacianフィルタ（シャープネス計算用）

### 学習目標
- [ ] Laplacianフィルタの概念を理解する
- [ ] `cv2.Laplacian()`の使い方を習得する

---

### 🔰 ざっくり説明（まずはここから）

**Laplacian（ラプラシアン）フィルタ**は、画像の「輪郭」や「境界線」を見つける道具です。

**ピントが合っている画像の特徴**:
- 物と物の境界がくっきりしている
- 色の変化が急激

**ピンぼけ画像の特徴**:
- 境界がぼんやりしている
- 色の変化がなだらか

Laplacianフィルタを使うと、この「くっきり度」を数値で測れます。

```
くっきりした画像 → Laplacian値が大きい → シャープネスが高い
ぼんやりした画像 → Laplacian値が小さい → シャープネスが低い
```

---

### 📚 詳しい説明

#### 2.3.1 Laplacianフィルタとは

**技術的な説明**:  
Laplacianフィルタは**2次微分（にじびぶん）** を計算します。

| 用語 | 簡単な説明 |
|------|------------|
| **微分** | 変化の度合いを調べること |
| **1次微分** | 「どれくらい変化しているか」 |
| **2次微分** | 「変化のスピードがどれくらい変わっているか」 |

画像で言うと：
- **1次微分** = 隣のピクセルとの明るさの差
- **2次微分** = その差がさらにどう変わっているか（エッジの鋭さ）

---

#### 2.3.2 シャープネス計算の実装

```python
import cv2
import numpy as np

# グレースケール画像を用意
gray = cv2.imread('sample.jpg', cv2.IMREAD_GRAYSCALE)

# Laplacianフィルタを適用
laplacian = cv2.Laplacian(gray, cv2.CV_64F)

# 分散を計算 → シャープネスの指標
sharpness = np.var(laplacian)
print(f"シャープネス: {sharpness}")
```

**コードの解説**:

| コード | 意味 |
|--------|------|
| `cv2.Laplacian(gray, cv2.CV_64F)` | Laplacianフィルタを適用 |
| `cv2.CV_64F` | 出力を64ビット浮動小数点数にする |
| `np.var(laplacian)` | 結果の分散を計算（Day 1で学習） |

---

#### ❓ なぜ分散を使うの？

Laplacianフィルタの出力は、エッジの強さを表す数値の配列です。

- **ピントが合った画像**: エッジがたくさんある → Laplacian値のバラつきが大きい → **分散が大きい**
- **ピンぼけ画像**: エッジが少ない → Laplacian値が全体的に小さい → **分散が小さい**

```python
# イメージ
sharp_image_laplacian = [-100, 200, -150, 300, ...]  # バラつき大
blurry_image_laplacian = [5, 10, 8, 12, ...]        # バラつき小
```

---

#### 🔢 cv2.CV_64Fについて

| 定数 | データ型 | 範囲 |
|------|----------|------|
| `cv2.CV_8U` | 8ビット符号なし整数 | 0〜255 |
| `cv2.CV_64F` | 64ビット浮動小数点数 | 非常に広い範囲 |

> 💡 **なぜCV_64Fを使う？**  
> Laplacianフィルタの結果は**マイナスの値**も含みます。
> `cv2.CV_8U`だとマイナスの値が0になってしまうので、`cv2.CV_64F`を使います。

---

### 💡 プロジェクトでの使用場面

- **Day 5**: `calc_sharpness()` 関数の実装

```python
# プロジェクトで作成する関数のイメージ
def calc_sharpness(image):
    """シャープネスを計算する"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    return np.var(laplacian)
```

---

### 📖 基礎事項の確認ポイント

| わからないこと | 確認する内容 |
|----------------|--------------|
| `np.var()` | Day 1のNumPy統計関数（分散） |
| グレースケール変換 | このDay 2のセクション2.2 |

---

### 🔗 公式リファレンス

- [cv2.Laplacian - OpenCV公式ドキュメント](https://docs.opencv.org/4.x/d4/d86/group__imgproc__filter.html#gad78703e4c8fe703d479c1860d76429e6)
- [Laplacian（概念説明）- OpenCV Tutorial](https://docs.opencv.org/4.x/d5/db5/tutorial_laplace_operator.html)

---

## 2.4 Gaussianぼかし（人工ピンぼけ生成用）

### 学習目標
- [ ] `cv2.GaussianBlur()`でぼかし処理ができる
- [ ] σ（シグマ）の意味を理解する

---

### 🔰 ざっくり説明（まずはここから）

**Gaussianぼかし（ガウシアンぼかし）** は、画像を「ぼかす」処理です。

スマホのカメラアプリで背景をぼかすような効果を、プログラムで作れます。

- **σ（シグマ）が小さい** → 少しだけぼける
- **σ（シグマ）が大きい** → 強くぼける

```
元画像 → GaussianBlur(σ=1) → 少しぼけた画像
元画像 → GaussianBlur(σ=5) → かなりぼけた画像
```

---

### 📚 詳しい説明

#### 2.4.1 Gaussianぼかしとは

**技術的な説明**:  
Gaussianぼかしは、**ガウス分布（正規分布）** という数学的な関数を使って画像をぼかします。

| 用語 | 簡単な説明 |
|------|------------|
| **ガウス分布** | 中心が一番高く、端に行くほど低くなる釣鐘型の形 |
| **σ（シグマ）** | ガウス分布の広がり具合を決める数値 |
| **カーネル** | ぼかしに使う数値の組み合わせ（小さな行列） |

---

#### 2.4.2 基本的な使い方

```python
import cv2

# 画像読み込み
image = cv2.imread('sample.jpg')

# 方法1: カーネルサイズを指定
blurred = cv2.GaussianBlur(image, (5, 5), 0)

# 方法2: σ（シグマ）を明示的に指定
blurred_sigma3 = cv2.GaussianBlur(image, (0, 0), sigmaX=3)
```

**コードの解説**:

| 引数 | 意味 | 例 |
|------|------|-----|
| 第1引数 | 入力画像 | `image` |
| 第2引数 | カーネルサイズ | `(5, 5)` または `(0, 0)` |
| 第3引数 | sigmaX（横方向のぼかし強度） | `0`（自動）または `3`（指定） |

---

#### 📋 パラメータの詳細

**カーネルサイズ**:
- `(5, 5)` のように**奇数×奇数**で指定
- 数字が大きいほどぼかしの範囲が広がる
- `(0, 0)` にすると、sigmaXから自動計算される

**sigmaX（σ）**:
- ぼかしの強さを決める値
- 大きいほど強くぼける
- `0` にすると、カーネルサイズから自動計算される

```python
# 推奨: sigmaXを明示的に指定する方法
blur_weak = cv2.GaussianBlur(image, (0, 0), sigmaX=1)   # 弱いぼかし
blur_medium = cv2.GaussianBlur(image, (0, 0), sigmaX=3)  # 中程度
blur_strong = cv2.GaussianBlur(image, (0, 0), sigmaX=5)  # 強いぼかし
```

---

#### 2.4.3 σ（シグマ）の効果を比較

```python
import cv2
import matplotlib.pyplot as plt

# 画像読み込み
image = cv2.imread('sample.jpg')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 異なるσでぼかし
sigma_values = [1, 3, 5, 10]
fig, axes = plt.subplots(1, 5, figsize=(15, 3))

# 元画像
axes[0].imshow(image_rgb)
axes[0].set_title("元画像")
axes[0].axis('off')

# 各σでぼかした画像
for i, sigma in enumerate(sigma_values):
    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=sigma)
    blurred_rgb = cv2.cvtColor(blurred, cv2.COLOR_BGR2RGB)
    axes[i+1].imshow(blurred_rgb)
    axes[i+1].set_title(f"σ = {sigma}")
    axes[i+1].axis('off')

plt.tight_layout()
plt.show()
```

---

#### 💡 シャープネスとの関係

ぼかし処理をすると、シャープネスが下がります。

```python
import cv2
import numpy as np

def calc_sharpness(image):
    """シャープネスを計算"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    return np.var(laplacian)

# 画像読み込み
image = cv2.imread('sample.jpg')

# 元画像のシャープネス
print(f"元画像: {calc_sharpness(image):.2f}")

# ぼかした画像のシャープネス
for sigma in [1, 3, 5]:
    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=sigma)
    print(f"σ={sigma}: {calc_sharpness(blurred):.2f}")

# 出力例:
# 元画像: 1250.00
# σ=1: 800.00
# σ=3: 150.00
# σ=5: 50.00
```

---

### 💡 プロジェクトでの使用場面

- **Day 8〜9**: 人工ピンぼけ画像の生成（テストデータ作成）

```python
# プロジェクトでの使用イメージ
def create_blurred_test_images(original_image):
    """テスト用のぼけ画像を生成"""
    results = []
    for sigma in [1, 2, 3, 4, 5]:
        blurred = cv2.GaussianBlur(original_image, (0, 0), sigmaX=sigma)
        results.append({
            'sigma': sigma,
            'image': blurred
        })
    return results
```

---

### 📖 基礎事項の確認ポイント

| わからないこと | 確認する内容 |
|----------------|--------------|
| 関数のキーワード引数 | Python関数の`sigmaX=3`のような書き方 |
| タプル `(5, 5)` | Pythonのタプルの基本 |

---

### 🔗 公式リファレンス

- [cv2.GaussianBlur - OpenCV公式ドキュメント](https://docs.opencv.org/4.x/d4/d86/group__imgproc__filter.html#gaabe8c836e97159a9193fb0b11ac52cf1)
- [Smoothing Images（ぼかしの概念説明）](https://docs.opencv.org/4.x/d4/d13/tutorial_py_filtering.html)

---

## 2.5 ✅ OpenCV確認問題

学習した内容を確認しましょう。コードの出力を予想してから実行してみてください。

### 問題1: shapeの変化

```python
import cv2

# カラー画像を読み込み（サイズ: 480×640）
color = cv2.imread('sample.jpg')  # shape: (480, 640, 3)

# グレースケールに変換
gray_image = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)

# gray_image.shape は何になる？
```

---

### 問題2: シャープネス計算

シャープネス計算の正しい順序はどれ？

- **A**: Laplacian → mean（平均）
- **B**: Laplacian → var（分散）
- **C**: GaussianBlur → var（分散）

---

### 問題3: ぼかしの強さ

ぼかしを強くするにはどうする？

- **A**: sigmaXを大きくする
- **B**: sigmaXを小さくする
- **C**: sigmaXは関係ない

---

### 問題4: 読み込みエラー

以下のコードで、存在しないファイルを読み込もうとしたとき、何が起こる？

```python
import cv2

image = cv2.imread('sonzai_shinai_file.jpg')
print(type(image))
```

- **A**: エラーが発生してプログラムが止まる
- **B**: `None`が返ってくる
- **C**: 空の配列が返ってくる

---

<details>
<summary>🔍 解答を見る</summary>

### 問題1の解答: `(480, 640)`

```
グレースケール変換で3チャンネル → 1チャンネルになる

カラー: (480, 640, 3) → 高さ480、幅640、3色
グレー: (480, 640)    → 高さ480、幅640のみ（色の情報がなくなる）
```

---

### 問題2の解答: **B（Laplacian → var）**

```
理由:
- Laplacianフィルタでエッジ（輪郭）を検出
- その結果の分散（var）でシャープネスを測る

Aが間違いの理由:
- 平均（mean）だとエッジの「平均的な強さ」になってしまう
- シャープネスは「バラつき」を見たいので分散を使う

Cが間違いの理由:
- GaussianBlurはぼかす処理
- シャープネス計算にはLaplacianを使う
```

---

### 問題3の解答: **A（sigmaXを大きくする）**

```
σ（シグマ）の値とぼかしの関係:
- σ = 1 → 弱いぼかし
- σ = 3 → 中程度のぼかし
- σ = 5 → 強いぼかし

σが大きいほど、より広い範囲のピクセルを混ぜ合わせるので、
強くぼける
```

---

### 問題4の解答: **B（`None`が返ってくる）**

```
cv2.imread()の重要な特徴:
- ファイルが存在しなくてもエラーにならない
- 代わりに None という特別な値を返す
- 必ず if image is None: でチェックする習慣をつけよう

print(type(image)) の出力は <class 'NoneType'> になる
```

</details>

---

## 📝 Day 2 まとめ

### 学んだこと

| 関数/概念 | 用途 |
|-----------|------|
| `cv2.imread()` | 画像の読み込み |
| `cv2.cvtColor()` | 色空間の変換（BGR↔グレースケール↔RGB） |
| `cv2.Laplacian()` | エッジ検出（シャープネス計算用） |
| `cv2.GaussianBlur()` | 画像のぼかし |

### 重要ポイント

1. **`cv2.imread()`は失敗してもエラーにならない** → 必ず`None`チェック
2. **OpenCVはBGR形式** → matplotlibで表示するときはRGBに変換
3. **シャープネス = Laplacianの分散** → `np.var(cv2.Laplacian(gray, cv2.CV_64F))`
4. **σが大きいほど強くぼける** → `cv2.GaussianBlur(image, (0, 0), sigmaX=σ)`

---

## 🚀 次のステップ

Day 3では、Pythonの関数定義とエラー処理を学びます。  
これにより、Day 2で学んだOpenCVの処理を**再利用可能な関数**として整理できるようになります。

---

## 📚 参考リンク集

- [OpenCV公式ドキュメント（英語）](https://docs.opencv.org/4.x/)
- [OpenCV-Python チュートリアル（英語）](https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html)
- [NumPy公式ドキュメント](https://numpy.org/doc/stable/)
