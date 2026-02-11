# Day15: Streamlit起動 + 画像アップロード + プレビュー（最小）
from __future__ import annotations
from pathlib import Path
import cv2
import numpy as np
import streamlit as st

def _decode_image(uploaded_file)-> np.ndarray:
    """StreamlitのUploadedFile（バイト列）をOpenCV画像（BGR）へデコードする"""
    data = uploaded_file.getvalue()
    arr = np.frombuffer(data, dtype=np.uint8)
    img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError("画像のデコードに失敗しました（対応形式か確認してください）")
    return img_bgr

def main()->None:
    st.set_page_config(page_title="Bioimage QC", layout="centered")
    st.title("Bioimage Quality Check")
    st.caption("Day15: Streamlit導入（起動 + 画像アップロード）")
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
    st.success("Day15完了：アップロードとプレビュー表示ができました。")


if __name__ == "__main__":
    main()