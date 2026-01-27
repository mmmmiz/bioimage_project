import argparse
from pathlib import Path
import cv2
import matplotlib.pyplot as plt

def load_bgr_image(path: str):
    """OpenCVで画像をBGRとして読み込む。失敗時はNone。"""
    img_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    return img_bgr

def bgr_to_rgb(img_bar):
    """matplotlib向けにBGR→RGBへ変換。"""
    return cv2.cvtColor(img_bar, cv2.COLOR_BGR2RGB)

def bgr_to_gray(img_bar):
    """BGR→グレースケールへ変換。"""
    return cv2.cvtColor(img_bar, cv2.COLOR_BGR2GRAY)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True, help="画像パス（例： data/sample.jpg)")
    args = parser.parse_args()
    
    path = Path(args.path)
    if not path.exists():
        raise FileNotFoundError(f"ファイルが見つかりません: {path}")
    
    # 1) 読み込み（最小動線）
    img_bgr = load_bgr_image(str(path))
    if img_bgr is None:
        raise ValueError("cv2.imread が失敗しました（対応していない形式 or パス不正の可能性）")
    print("=== Color(BGR) ===")
    print("shape:", img_bgr.shape)  # (H, W, 3) 
    print("shape:", img_bgr.dtype) # uint8 が多い
    
    # 2) カラー表示（BGR→RGB）
    img_rgb = bgr_to_rgb(img_bgr)
    plt.figure()
    plt.title("Color (RGB)")
    plt.imshow(img_rgb)
    plt.axis("off")
    plt.show()
    
    # 3) グレースケール変換＋表示
    img_gray = bgr_to_gray(img_bgr)
    print("=== Grayscale ===")
    print("shape:", img_gray.shape)  # (H, W) のはず
    print("dtype:", img_gray.dtype)
    plt.figure()
    plt.title("Grayscale")
    plt.imshow(img_gray, cmap="gray")
    plt.axis("off")
    plt.show()
    
    # 4) （理解固め）B/G/R のどれか1チャンネルを表示
    # OpenCVのBGR順なので、0=B,1=G,2=R
    b_channel = img_bgr[:, :, 0]
    plt.figure()
    plt.title("B channel (from BGR)")
    plt.imshow(b_channel, cmap="gray")
    plt.axis("off")
    plt.show()
    
if __name__ == "__main__":
    main()
    
"""
OpenCV : BGR
matplotlib : RGB

"""
"""imread()関数
imread()は画像をNumpyの配列として返す

img = cv2.imread('image.jpg')
返される配列は各ピクセルのBGR値を含む3次元Numpy配列 （ 読み込めない ⇨ None ）

【オプション】
① cv2.IMREAD_COLOR
    カラー画像として読み込む
② cv2.IMREAD_GRAYSCALE
    グレースケール画像として読み込む
③ cv2.IMREAD_UNCHANGED
    そのままの形式で読み込む（透過、複数チャンネルもつとき）
"""
"""cvtColor関数
cv2.cvtColor(img, code) 

codeには画像変換方法を入れる（ColorConversionCodes）
COLOR_BGR2BGRA ： BGR から BGRA への拡張変換。
COLOR_BGR2GRAY ： BGR から グレースケールへ変換。
COLOR_RGBA2BGR ： RGB から BGR へのドットの並び方変換。     などなど

"""
"""ArgumentParser (argparse)
Pythonの実行時にコマンドライン引数を取りたいとき
https://qiita.com/kzkadc/items/e4fc7bc9c003de1eb6d0
"""