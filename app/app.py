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

def _build_thresholds_ui() -> dict:
    """Day17: しきい値をスライダーで入力して dict にまとめる"""
    st.sidebar.header("Thresholds")
    b_min, b_max = st.sidebar.slider(
        "brightness_mean(min, max)",
        min_value=0,
        max_value=255,
        value=(50, 220),
        step=1,
        key="th_brightness_range",
    )
    c_min, c_max = st.sidebar.slider(
        "contrast_std(min, max)",
        min_value=0,
        max_value=128,
        value=(10, 80),
        step=1,
        key="th_contrast_range",
    )
    s_min = st.sidebar.slider(
        "sharpness_lap_var(min)",
        min_value=0,
        max_value=50000,
        value=300,
        step=10,
        key="th_sharpness_mean",
    )
    
    thresholds = {
        "brightness_mean": {"min": float(b_min),"max": float(b_max)},
        "contrast_std":{"min": float(c_min),"max":float(c_max)},
        "sharpness_lap_var":{"min": float(s_min)},
    }
    return thresholds

def main()->None:
    st.set_page_config(page_title="Bioimage QC", layout="centered")
    st.title("Bioimage Quality Check")
    st.caption("Day17: しきい値（thresholds）をUI化")
    st.write("このアプリは 明るさ・コントラスト・シャープネス の3指標で画像品質をチェックし、しきい値に基づいて OK/NG を判定します。")
    # サイドバーでしきい値を入力
    thresholds = _build_thresholds_ui()
    # しきい値を確認できるようにしておく（デバッグ用）
    with st.expander("Current thresholds(debug)",expanded=False):
        st.json(thresholds)
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
        # metrics と thresholds の比較プレビュー（Day18の準備）
        with st.expander("Metrics vs thresholds(preview)",expanded=False):
            st.write("Day18でこのthresholdsを使ってOK/NG判定を表示します。")
            st.json({"metrics": metrics,"thresholds":thresholds})
    finally:
        try:
            tmp_path.unlink()
        except Exception:
            pass
    st.success("Day17完了：しきい値をスライダーで調整できました。")

if __name__ == "__main__":
    main()