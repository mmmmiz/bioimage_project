# Day 3: Python関数・エラー処理（1〜2時間）

---

## 📋 今日の学習内容

| セクション | 内容 | 所要時間目安 |
|-----------|------|-------------|
| 3.1 | 関数の定義と戻り値 | 30〜40分 |
| 3.2 | 例外処理（try-except） | 30〜40分 |
| 3.3 | パス操作（pathlib） | 20〜30分 |
| 3.4 | 確認問題 | 10〜20分 |

### 💡 プロジェクトでの使用場面（先に知っておこう）

今日学ぶ内容は、プロジェクトの以下の場面で使います：

- **関数定義** → Day 3〜5で `calc_brightness()`, `calc_contrast()`, `calc_sharpness()` などの指標計算関数を作る
- **例外処理** → Day 7でエラーハンドリング、Day 20でUI安定化
- **パス操作** → Day 1でディレクトリ構造設定、Day 9でデータセット作成時のフォルダ整理

---

## 3.1 関数の定義と戻り値

### 🎯 学習目標

- [ ] 引数（ひきすう）と戻り値（もどりち）を持つ関数が書ける
- [ ] 型ヒント（Type Hints）の書き方を理解する

---

### 📖 ざっくり説明（まずはここから）

**関数とは？**

関数は「材料を入れると、加工して結果を返してくれる機械」のようなものです。

```
【例え】ジュースミキサー
┌─────────────────────────────────────┐
│  材料（引数）    →  ミキサー（関数）  →  ジュース（戻り値）  │
│  りんご、水      →  混ぜる処理        →  りんごジュース      │
└─────────────────────────────────────┘
```

プログラミングでは、同じ処理を何度も書かなくていいように、処理をまとめて名前をつけたものが「関数」です。

**型ヒントとは？**

型ヒントは「この関数には○○型のデータを入れてね」「返すのは△△型だよ」と書いておくメモのようなものです。書かなくても動きますが、書いておくとミスを防げます。

---

### 📚 詳細説明（しっかり理解する）

#### 3.1.1 基本的な関数定義

```python
import numpy as np

def calc_brightness(image):
    """画像の平均輝度を計算する"""
    return np.mean(image)
```

**コードの解説：**

| 部分 | 意味 |
|------|------|
| `def` | 「関数を定義します」という宣言（definition の略） |
| `calc_brightness` | 関数の名前（自分で決める） |
| `(image)` | 引数（ひきすう）＝関数に渡すデータ |
| `"""..."""` | docstring（ドキュメント文字列）＝関数の説明文 |
| `return` | 「この値を返します」という命令 |
| `np.mean(image)` | 戻り値（もどりち）＝関数が返す結果 |

**使用例：**

```python
import numpy as np

# 関数を定義
def calc_brightness(image):
    """画像の平均輝度を計算する"""
    return np.mean(image)

# 関数を使う
test_image = np.array([[100, 150], [200, 250]])
result = calc_brightness(test_image)  # 関数を呼び出す
print(result)  # 175.0
```

---

#### 3.1.2 型ヒント付きの関数（推奨）

実務では、型ヒントをつけることが推奨されています。

```python
import numpy as np
from numpy.typing import NDArray

def calc_brightness(image: NDArray[np.uint8]) -> float:
    """
    画像の平均輝度を計算する
    
    Parameters
    ----------
    image : NDArray[np.uint8]
        入力画像（グレースケール）
    
    Returns
    -------
    float
        平均輝度値（0.0〜255.0）
    """
    return float(np.mean(image))
```

**コードの解説：**

| 部分 | 意味 |
|------|------|
| `image: NDArray[np.uint8]` | 「imageはNumPy配列で、中身はuint8型だよ」という型ヒント |
| `-> float` | 「この関数はfloat型の値を返すよ」という型ヒント |
| `float(np.mean(image))` | 明示的にfloat型に変換（型ヒントと実際の型を一致させる） |

**専門用語の解説：**

| 用語 | 意味 | 平易な言い換え |
|------|------|---------------|
| 引数（ひきすう）| 関数に渡すデータ | 関数への入力、材料 |
| 戻り値（もどりち）| 関数が返すデータ | 関数からの出力、結果 |
| 型ヒント（Type Hints）| 引数や戻り値のデータ型を示す注釈 | 「この型を使ってね」というメモ |
| docstring | 関数の説明を書く文字列 | 関数の取扱説明書 |
| NDArray | NumPyの配列型を表す型ヒント用の型 | NumPy配列であることを示す記号 |
| uint8 | 0〜255の整数を表すデータ型 | 画像のピクセル値に使う型 |

---

#### 3.1.3 複数の引数を持つ関数

```python
def is_brightness_ok(brightness: float, threshold: float = 100.0) -> bool:
    """
    明るさがしきい値以上かどうかを判定する
    
    Parameters
    ----------
    brightness : float
        計算された明るさの値
    threshold : float, optional
        しきい値（デフォルト: 100.0）
    
    Returns
    -------
    bool
        しきい値以上ならTrue、未満ならFalse
    """
    return brightness >= threshold

# 使用例
print(is_brightness_ok(120.0))        # True（デフォルトのしきい値100を使用）
print(is_brightness_ok(80.0))         # False
print(is_brightness_ok(80.0, 70.0))   # True（しきい値を70に変更）
```

**ポイント：**

- `threshold: float = 100.0` のように書くと、引数を省略したときのデフォルト値を設定できます
- デフォルト値がある引数は、関数を呼ぶときに省略できます

---

### 🔍 基礎事項の確認先

この内容がわからない場合は、以下を確認してください：

| わからない点 | 確認すべき基礎事項 |
|-------------|-------------------|
| `def` の使い方がわからない | Python公式: [関数の定義](https://docs.python.org/ja/3/tutorial/controlflow.html#defining-functions) |
| `return` の意味がわからない | Python公式: [return文](https://docs.python.org/ja/3/reference/simple_stmts.html#the-return-statement) |
| 型ヒントの書き方 | Python公式: [型ヒント](https://docs.python.org/ja/3/library/typing.html) |
| NumPy配列がわからない | Day 1のNumPy基礎を復習 |

### 🔗 公式リファレンス

- [Python公式: 関数の定義](https://docs.python.org/ja/3/tutorial/controlflow.html#defining-functions)
- [Python公式: typing — 型ヒントのサポート](https://docs.python.org/ja/3/library/typing.html)
- [NumPy公式: numpy.typing.NDArray](https://numpy.org/doc/stable/reference/typing.html#numpy.typing.NDArray)

---

## 3.2 例外処理（try-except）

### 🎯 学習目標

- [ ] 基本的なtry-except文が書ける
- [ ] 画像読み込みエラーを適切に処理できる

---

### 📖 ざっくり説明（まずはここから）

**例外処理とは？**

プログラムは実行中にエラー（例外）が起きることがあります。例外処理は「エラーが起きたときにどうするか」を事前に決めておく仕組みです。

```
【例え】料理中のトラブル対応
┌─────────────────────────────────────────┐
│  try:（やってみる）                       │
│      卵を割る                             │
│  except 殻が入った:（もし失敗したら）      │
│      殻を取り除く                         │
│  except 卵が腐っていた:                   │
│      新しい卵を使う                       │
└─────────────────────────────────────────┘
```

例外処理がないと、エラーが起きた瞬間にプログラムが止まってしまいます。例外処理を書いておくと、エラーが起きても適切に対応してプログラムを続けられます。

---

### 📚 詳細説明（しっかり理解する）

#### 3.2.1 基本構文

```python
try:
    # エラーが起きる可能性のある処理
    result = risky_operation()
except Exception as e:
    # エラー時の処理
    print(f"エラーが発生しました: {e}")
```

**コードの解説：**

| 部分 | 意味 |
|------|------|
| `try:` | 「この中の処理を試してみる」というブロックの開始 |
| `except Exception as e:` | 「Exceptionが起きたら、それを `e` という変数に入れてここを実行」 |
| `Exception` | すべてのエラー（例外）の親クラス。これを指定するとどんなエラーでもキャッチできる |
| `as e` | 発生したエラーの情報を `e` という変数に格納する |

**実行の流れ：**

```
tryブロックを実行
    ↓
エラーなし → tryブロック終了後、exceptをスキップして次へ
    ↓
エラーあり → tryブロックを中断し、exceptブロックを実行
```

---

#### 3.2.2 画像読み込みでの実践例

OpenCVの `cv2.imread()` は、読み込みに失敗しても**エラーを出さずに `None` を返す**という特殊な動作をします。これを適切に処理する例を見てみましょう。

```python
import cv2

def load_image(filepath: str):
    """
    画像を安全に読み込む
    
    Parameters
    ----------
    filepath : str
        画像ファイルのパス
    
    Returns
    -------
    numpy.ndarray or None
        読み込んだ画像。失敗した場合はNone
    """
    try:
        # 画像を読み込む
        image = cv2.imread(filepath)
        
        # cv2.imread()は失敗してもエラーを出さずNoneを返す
        # なので、明示的にチェックしてエラーを発生させる
        if image is None:
            raise ValueError(f"画像を読み込めませんでした: {filepath}")
        
        return image
    
    except ValueError as e:
        # ValueErrorをキャッチして処理
        print(f"エラー: {e}")
        return None
    
    except Exception as e:
        # その他の予期しないエラーをキャッチ
        print(f"予期しないエラー: {e}")
        return None
```

**コードの解説：**

| 部分 | 意味 |
|------|------|
| `if image is None:` | 画像が読み込めなかった場合のチェック |
| `raise ValueError(...)` | 明示的にValueErrorを発生させる（投げる） |
| `except ValueError as e:` | ValueErrorだけをキャッチする |
| `except Exception as e:` | その他すべてのエラーをキャッチする（最後に書く） |

**なぜ `raise` で明示的にエラーを発生させるのか？**

```python
# cv2.imread()の困った動作
image = cv2.imread("存在しないファイル.jpg")
print(image)  # None が出力される（エラーにならない！）
print(image.shape)  # ここでエラー！NoneTypeにshapeはない

# 明示的にエラーを発生させると、どこで問題が起きたかわかりやすい
```

---

#### 3.2.3 複数の例外を個別に処理する

```python
def process_image(filepath: str) -> dict:
    """画像を処理して結果を返す"""
    try:
        # ファイルを開く
        image = cv2.imread(filepath)
        if image is None:
            raise FileNotFoundError(f"ファイルが見つかりません: {filepath}")
        
        # 処理を実行
        brightness = np.mean(image)
        
        return {"status": "success", "brightness": brightness}
    
    except FileNotFoundError as e:
        # ファイルが見つからない場合
        print(f"ファイルエラー: {e}")
        return {"status": "error", "message": "ファイルが見つかりません"}
    
    except MemoryError as e:
        # メモリ不足の場合
        print(f"メモリエラー: {e}")
        return {"status": "error", "message": "メモリ不足です"}
    
    except Exception as e:
        # その他のエラー
        print(f"予期しないエラー: {e}")
        return {"status": "error", "message": str(e)}
```

**ポイント：**

- 特定のエラーから順番に書き、最後に `Exception` を書く
- エラーの種類によって異なる対応ができる

---

#### 3.2.4 finally（必ず実行する処理）

```python
def process_with_cleanup():
    """処理後に必ずクリーンアップを行う"""
    try:
        # 何かの処理
        result = some_operation()
        return result
    except Exception as e:
        print(f"エラー: {e}")
        return None
    finally:
        # エラーがあってもなくても、必ず実行される
        print("クリーンアップ処理を実行")
        cleanup()
```

**専門用語の解説：**

| 用語 | 意味 | 平易な言い換え |
|------|------|---------------|
| 例外（Exception）| プログラム実行中に発生するエラー | 予期しない問題 |
| try | エラーが起きるかもしれない処理を囲むブロック | 「試してみる」ゾーン |
| except | エラーが起きたときの処理を書くブロック | 「エラー時はこうする」ゾーン |
| raise | 明示的にエラーを発生させる命令 | 「エラーを投げる」 |
| finally | エラーの有無に関わらず必ず実行されるブロック | 「最後に必ずやる」ゾーン |
| ValueError | 値が不正な場合のエラー | 「その値はおかしい」エラー |
| FileNotFoundError | ファイルが見つからない場合のエラー | 「ファイルがない」エラー |

---

### 🔍 基礎事項の確認先

この内容がわからない場合は、以下を確認してください：

| わからない点 | 確認すべき基礎事項 |
|-------------|-------------------|
| try-exceptの基本 | Python公式: [エラーと例外](https://docs.python.org/ja/3/tutorial/errors.html) |
| `None` の扱い | Python公式: [None型](https://docs.python.org/ja/3/library/constants.html#None) |
| `is` と `==` の違い | Python公式: [比較](https://docs.python.org/ja/3/reference/expressions.html#comparisons) |
| cv2.imread()の動作 | Day 2のOpenCV基礎を復習 |

### 🔗 公式リファレンス

- [Python公式: エラーと例外](https://docs.python.org/ja/3/tutorial/errors.html)
- [Python公式: 組み込み例外](https://docs.python.org/ja/3/library/exceptions.html)

---

## 3.3 パス操作（pathlib）

### 🎯 学習目標

- [ ] `pathlib.Path` でファイルパスを扱える
- [ ] 相対パス・絶対パスの違いを理解する

---

### 📖 ざっくり説明（まずはここから）

**pathlibとは？**

`pathlib` は、ファイルやフォルダの場所（パス）を扱うための便利なライブラリです。

昔のPythonでは文字列でパスを操作していましたが、`pathlib` を使うとより直感的に、OSの違い（Windows/Mac/Linux）を気にせずパスを扱えます。

```
【例え】住所の書き方
┌─────────────────────────────────────────┐
│  昔の方法（文字列）:                      │
│      "data" + "/" + "images" + "/" + "sample.jpg"  │
│      → OS によって "/" か "\" か変わる   │
│                                          │
│  pathlib の方法:                          │
│      Path("data") / "images" / "sample.jpg"  │
│      → OS を気にしなくてOK              │
└─────────────────────────────────────────┘
```

---

### 📚 詳細説明（しっかり理解する）

#### 3.3.1 基本的な使い方

```python
from pathlib import Path

# パスの作成
data_dir = Path("data")
image_path = data_dir / "sample.jpg"  # "/" でパスを連結できる

print(image_path)  # data/sample.jpg（Macの場合）

# パスの存在確認
if image_path.exists():
    print("ファイルが存在します")
else:
    print("ファイルが存在しません")

# ファイル名・拡張子の取得
print(image_path.name)    # sample.jpg（ファイル名全体）
print(image_path.stem)    # sample（拡張子を除いたファイル名）
print(image_path.suffix)  # .jpg（拡張子）
print(image_path.parent)  # data（親ディレクトリ）
```

**コードの解説：**

| 部分 | 意味 |
|------|------|
| `Path("data")` | "data" という文字列からPathオブジェクトを作成 |
| `data_dir / "sample.jpg"` | パスを `/` 演算子で連結（OSに合わせて自動で区切り文字を決める） |
| `.exists()` | ファイルやフォルダが存在するか確認 |
| `.name` | ファイル名（拡張子含む） |
| `.stem` | ファイル名（拡張子なし）|
| `.suffix` | 拡張子（ドット含む） |
| `.parent` | 親ディレクトリのパス |

---

#### 3.3.2 相対パスと絶対パス

```python
from pathlib import Path

# 相対パス：今いる場所からの道順
relative_path = Path("data/images/sample.jpg")
print(relative_path)  # data/images/sample.jpg

# 絶対パス：ルート（最上位）からの完全な道順
absolute_path = Path("/Users/username/project/data/images/sample.jpg")
print(absolute_path)  # /Users/username/project/data/images/sample.jpg

# 相対パスを絶対パスに変換
full_path = relative_path.resolve()
print(full_path)  # /Users/username/current_dir/data/images/sample.jpg

# 絶対パスかどうかを確認
print(relative_path.is_absolute())  # False
print(absolute_path.is_absolute())  # True
```

**相対パスと絶対パスの違い：**

| 種類 | 説明 | 例 |
|------|------|-----|
| 相対パス | 現在の作業ディレクトリからの相対的な位置 | `data/sample.jpg` |
| 絶対パス | ルートからの完全な位置 | `/Users/name/project/data/sample.jpg` |

---

#### 3.3.3 プロジェクトルートの設定

実際のプロジェクトでは、「プロジェクトのルートフォルダ」を基準にパスを設定することが多いです。

**Pythonスクリプト（.py）の場合：**

```python
from pathlib import Path

# __file__ は現在のスクリプトファイルのパス
# .parent で親ディレクトリを取得
# 例：scripts/analysis.py から見て、プロジェクトルートは2階層上

BASE_DIR = Path(__file__).parent.parent  # プロジェクトルート
DATA_DIR = BASE_DIR / "data"             # data フォルダ
OUTPUT_DIR = BASE_DIR / "output"         # output フォルダ

print(f"プロジェクトルート: {BASE_DIR}")
print(f"データフォルダ: {DATA_DIR}")
```

**Jupyter Notebookの場合：**

```python
from pathlib import Path

# Notebookでは __file__ が使えないので、手動で設定する
# または、Notebookの場所を基準にする

# 方法1: 手動で設定
BASE_DIR = Path("/Users/username/project")

# 方法2: カレントディレクトリを基準にする
import os
BASE_DIR = Path(os.getcwd())

DATA_DIR = BASE_DIR / "data"
```

---

#### 3.3.4 ディレクトリの作成と一覧取得

```python
from pathlib import Path

# ディレクトリを作成
output_dir = Path("output/results")
output_dir.mkdir(parents=True, exist_ok=True)
# parents=True: 親ディレクトリも一緒に作る
# exist_ok=True: すでに存在してもエラーにしない

# ディレクトリ内のファイル一覧を取得
data_dir = Path("data")

# すべてのファイル・フォルダを取得
for item in data_dir.iterdir():
    print(item)

# 特定の拡張子のファイルだけ取得（glob）
for jpg_file in data_dir.glob("*.jpg"):
    print(jpg_file)

# サブフォルダも含めて検索（再帰的glob）
for jpg_file in data_dir.glob("**/*.jpg"):
    print(jpg_file)
```

**コードの解説：**

| 部分 | 意味 |
|------|------|
| `.mkdir()` | ディレクトリを作成 |
| `parents=True` | 親ディレクトリが存在しなければ一緒に作成 |
| `exist_ok=True` | すでに存在してもエラーにしない |
| `.iterdir()` | ディレクトリ内の全アイテムをイテレート（繰り返し処理） |
| `.glob("*.jpg")` | パターンにマッチするファイルを検索 |
| `**/*.jpg` | すべてのサブフォルダを含めて `.jpg` を検索 |

---

#### 3.3.5 プロジェクトでの実践例

```python
from pathlib import Path
import cv2

def setup_directories(base_dir: Path) -> dict:
    """
    プロジェクトのディレクトリ構造をセットアップする
    
    Parameters
    ----------
    base_dir : Path
        プロジェクトのルートディレクトリ
    
    Returns
    -------
    dict
        各ディレクトリのパスを格納した辞書
    """
    dirs = {
        "data": base_dir / "data",
        "output": base_dir / "output",
        "logs": base_dir / "logs",
    }
    
    # 各ディレクトリを作成
    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    
    return dirs


def get_image_files(data_dir: Path) -> list:
    """
    データディレクトリから画像ファイルのリストを取得する
    
    Parameters
    ----------
    data_dir : Path
        画像が格納されているディレクトリ
    
    Returns
    -------
    list
        画像ファイルのPathオブジェクトのリスト
    """
    # 対応する拡張子
    extensions = ["*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff"]
    
    image_files = []
    for ext in extensions:
        image_files.extend(data_dir.glob(ext))
    
    return sorted(image_files)  # ファイル名でソート


# 使用例
if __name__ == "__main__":
    BASE_DIR = Path(".")  # カレントディレクトリ
    
    dirs = setup_directories(BASE_DIR)
    print(f"データフォルダ: {dirs['data']}")
    
    images = get_image_files(dirs["data"])
    print(f"画像ファイル数: {len(images)}")
    
    for img_path in images:
        print(f"  - {img_path.name}")
```

**専門用語の解説：**

| 用語 | 意味 | 平易な言い換え |
|------|------|---------------|
| パス（Path）| ファイルやフォルダの場所を示す文字列 | ファイルの住所 |
| 相対パス | 現在地からの相対的な場所 | 「ここから右に2軒目」のような指定 |
| 絶対パス | ルートからの完全な場所 | 「○○市△△町1-2-3」のような指定 |
| ディレクトリ | フォルダのこと | ファイルを入れる箱 |
| glob | ワイルドカードを使ったファイル検索 | 「*.jpg」で「すべてのjpgファイル」を検索 |
| イテレート | 繰り返し処理すること | 一つずつ順番に見ていくこと |

---

### 🔍 基礎事項の確認先

この内容がわからない場合は、以下を確認してください：

| わからない点 | 確認すべき基礎事項 |
|-------------|-------------------|
| Pathオブジェクトの基本 | Python公式: [pathlib](https://docs.python.org/ja/3/library/pathlib.html) |
| for文の使い方 | Python公式: [for文](https://docs.python.org/ja/3/tutorial/controlflow.html#for-statements) |
| リストの操作 | Python公式: [リスト](https://docs.python.org/ja/3/tutorial/datastructures.html#more-on-lists) |
| `__file__` の意味 | Python公式: [__file__](https://docs.python.org/ja/3/reference/import.html#file__) |

### 🔗 公式リファレンス

- [Python公式: pathlib — オブジェクト指向のファイルシステムパス](https://docs.python.org/ja/3/library/pathlib.html)
- [Python公式: pathlib チュートリアル](https://docs.python.org/ja/3/library/pathlib.html#basic-use)

---

## 3.4 ✅ 確認問題

以下の問題を解いて、理解度をチェックしましょう。

### 問題1: 関数の戻り値の型

```python
import numpy as np

def calc_contrast(image) -> float:
    return np.std(image)

# この関数の戻り値の型は何でしょうか？
```

### 問題2: 例外処理の動作

```python
import cv2

try:
    image = cv2.imread("not_exist.jpg")
    if image is None:
        raise ValueError("読み込み失敗")
    print("成功")
except ValueError as e:
    print(f"キャッチ: {e}")

# 上のコードを実行すると、何が出力されるでしょうか？
```

### 問題3: パス操作

```python
from pathlib import Path

result = Path("data") / "images" / "sample.jpg"

# resultの値は何でしょうか？
```

### 問題4: ファイル名の取得

```python
from pathlib import Path

filepath = Path("/Users/name/project/data/image001.png")

# 以下の各属性の値を答えてください
# filepath.name → ?
# filepath.stem → ?
# filepath.suffix → ?
# filepath.parent → ?
```

### 問題5: 実践問題

以下の要件を満たす関数 `load_images_from_dir` を作成してください。

要件：
- 引数：ディレクトリパス（Path型）
- 戻り値：読み込んだ画像のリスト（list型）
- 動作：指定したディレクトリ内の `.jpg` ファイルをすべて読み込む
- エラー処理：読み込みに失敗した画像はスキップする

---

<details>
<summary>📝 解答を見る</summary>

### 問題1の解答

```
float（型ヒントで -> float と明示されている）
```

### 問題2の解答

```
キャッチ: 読み込み失敗

解説：
1. cv2.imread("not_exist.jpg") は存在しないファイルなので None を返す
2. if image is None: が True になる
3. raise ValueError("読み込み失敗") でValueErrorが発生
4. except ValueError as e: でキャッチされる
5. print(f"キャッチ: {e}") が実行される
```

### 問題3の解答

```
Path("data/images/sample.jpg")
または文字列として表示すると "data/images/sample.jpg"

解説：
pathlib.Pathでは / 演算子でパスを連結できます。
OSによって実際の区切り文字は自動で調整されます。
（Macでは /、Windowsでは \）
```

### 問題4の解答

```
filepath.name   → "image001.png"（ファイル名全体）
filepath.stem   → "image001"（拡張子を除いたファイル名）
filepath.suffix → ".png"（拡張子、ドット含む）
filepath.parent → Path("/Users/name/project/data")（親ディレクトリ）
```

### 問題5の解答

```python
from pathlib import Path
import cv2
from typing import List
import numpy as np

def load_images_from_dir(dir_path: Path) -> List[np.ndarray]:
    """
    ディレクトリ内のJPG画像をすべて読み込む
    
    Parameters
    ----------
    dir_path : Path
        画像が格納されているディレクトリのパス
    
    Returns
    -------
    List[np.ndarray]
        読み込んだ画像のリスト
    """
    images = []
    
    # .jpg ファイルを検索
    for jpg_file in dir_path.glob("*.jpg"):
        try:
            # 画像を読み込み
            image = cv2.imread(str(jpg_file))
            
            # 読み込み失敗のチェック
            if image is None:
                print(f"スキップ（読み込み失敗）: {jpg_file.name}")
                continue
            
            images.append(image)
            print(f"読み込み成功: {jpg_file.name}")
            
        except Exception as e:
            print(f"スキップ（エラー）: {jpg_file.name} - {e}")
            continue
    
    return images


# 使用例
if __name__ == "__main__":
    data_dir = Path("data")
    loaded_images = load_images_from_dir(data_dir)
    print(f"読み込んだ画像数: {len(loaded_images)}")
```

**ポイント：**
- `glob("*.jpg")` で `.jpg` ファイルだけを検索
- `str(jpg_file)` で Path を文字列に変換（cv2.imread は文字列を受け取る）
- 読み込み失敗時は `continue` でスキップ
- try-except で予期しないエラーにも対応

</details>

---

## 📝 今日のまとめ

### 学んだこと

1. **関数の定義**
   - `def 関数名(引数): return 戻り値` の形式
   - 型ヒントで引数と戻り値の型を明示できる
   - docstringで関数の説明を書く

2. **例外処理**
   - `try-except` でエラーを安全に処理できる
   - `raise` で明示的にエラーを発生させられる
   - `cv2.imread()` は失敗しても `None` を返すだけなので注意

3. **パス操作**
   - `pathlib.Path` でOSを気にせずパスを扱える
   - `/` 演算子でパスを連結できる
   - `.exists()`, `.glob()`, `.mkdir()` などの便利なメソッド

### チェックリスト

- [ ] 型ヒント付きの関数を定義できる
- [ ] try-exceptでエラーを処理できる
- [ ] raiseで明示的にエラーを発生させられる
- [ ] pathlibでパスを扱える
- [ ] globでファイルを検索できる

---

## 🚀 次のステップ

Day 3の学習お疲れさまでした！

次はDay 4「Streamlit基礎」で、実際にWebアプリのUIを作る方法を学びます。
今日学んだ関数定義やエラー処理は、Day 3〜5で指標計算関数を実装するときに使います。

わからない点があれば、いつでも質問してください！
