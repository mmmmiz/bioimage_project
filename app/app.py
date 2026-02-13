from __future__ import annotations
from pathlib import Path
import cv2
import numpy as np
import streamlit as st
import sys
import tempfile
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))
    
from bioimage_qc.pipeline import evaluate_image
    
def _decode_image(uploaded_file)-> np.ndarray:
    """StreamlitのUploadedFile（バイト列）をOpenCV画像（BGR）へデコードする"""
    data = uploaded_file.getvalue()
    arr = np.frombuffer(data, dtype=np.uint8)
    img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError("画像のデコードに失敗しました（対応形式か確認してください）")
    return img_bgr

def _save_to_temp(uploaded_file) -> Path:
    """UploadedFileを一時ファイルに保存し、そのPathを返す"""
    suffix = Path(uploaded_file.name).suffix.lower() or ".png"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
        f.write(uploaded_file.getvalue())
        return Path(f.name)

def main()->None:
    st.set_page_config(page_title="Bioimage QC", layout="centered")
    st.title("Bioimage Quality Check")
    st.caption("Day16: 指標（metrics）を表で表示")
    # Day14で作った短文をここに貼る想定（今は仮）
    st.write("このアプリは 明るさ・コントラスト・シャープネス の3指標で画像品質をチェックし、しきい値に基づいて OK/NG を判定します。")
    uploaded = st.file_uploader(
        label="画像ファイル（jpg/png）を選択してください", type=["jpg", "jpeg", "png"],
        accept_multiple_files=False,
    )
    
    if uploaded is None:
        st.info("画像をアップロードすると、ここにプレビューが表示されます。")
        return
    # アップロード情報（軽いメタ情報）
    st.write(
        {
            "filename":uploaded.name,
            "type":uploaded.type,
            "size_bytes":uploaded.size,
        }
    )
    # 画像デコード → RGBに変換してプレビュー
    img_bgr = _decode_image(uploaded)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    st.subheader("Preview")
    st.image(img_rgb, caption=uploaded.name, use_container_width=True)
    
    # 指標を計算して表表示
    st.subheader("Metrics")
    tmp_path = _save_to_temp(uploaded)
    try:
        result = evaluate_image(tmp_path)
        metrics = result.get("metrics", {})
        
        rows = [{"metric": k, "value": float(v)} for k, v in metrics.items()]
        df = pd.DataFrame(rows)
        df["value"] = df["value"].map(lambda x:round(x, 4))
        st.dataframe(df,use_container_width=True)
    finally:
        try:
            tmp_path.unlink()
        except Exception:
            pass
    st.success("Day16完了：指標を表で表示できました。")

if __name__ == "__main__":
    main()