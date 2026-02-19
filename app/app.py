# app/app.py
# OK/NG判定（judgement）を表示（修正）

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
    """しきい値をスライダーで入力して dict にまとめる"""
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

def _render_judgement(judgement: dict) -> None:
    """Day18: judgementを見やすく表示する"""
    ok = bool(judgement.get("ok", False))
    label = judgement.get("label", "OK" if ok else "NG")
    reasons = judgement.get("reasons", [])

    st.subheader("Judgement")

    if ok:
        st.success(f"{label}")
        if reasons:
            st.caption("Notes")
            for r in reasons:
                st.write(f"- {r}")
    else:
        st.error(f"{label}")
        if reasons:
            st.caption("Reasons")
            for r in reasons:
                st.write(f"- {r}")
        else:
            st.write("理由が取得できませんでした（judgeの返り値を確認してください）。")
def main()->None:
    st.set_page_config(page_title="Bioimage QC", layout="centered")
    st.title("Bioimage Quality Check")
    st.caption("Day18: OK/NG判定を表示（理由付き）")
    st.write("このアプリは 明るさ・コントラスト・シャープネス の3指標で画像品質をチェックし、しきい値に基づいて OK/NG を判定します。")

    thresholds = _build_thresholds_ui()
    with st.expander("Current thresholds(debug)",expanded=False):
        st.json(thresholds)
        
    uploaded = st.file_uploader(
        label="画像ファイル（jpg/png）を選択してください", type=["jpg", "jpeg", "png"],
        accept_multiple_files=False,
    )
    
    if uploaded is None:
        st.info("画像をアップロードすると、ここにプレビューが表示されます。")
        return

    # Preview
    img_bgr = _decode_image(uploaded)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    st.subheader("Preview")
    st.image(img_rgb, caption=uploaded.name, use_container_width=True)
    
    # Evaluate (metrics + judgement)
    tmp_path = _save_to_temp(uploaded)
    ok = False
    try:
        # ★ Day18: thresholdsを渡す（pipelineが対応している前提）
        result = evaluate_image(tmp_path, thresholds=thresholds)
        metrics = result.get("metrics", {})
        judgement = result.get("judgement", {})
        
        # Metrics 表
        st.subheader("Metrics")
        rows = [{"metric": k, "value": float(v)} for k, v in metrics.items()]
        df = pd.DataFrame(rows)
        df["value"] = df["value"].map(lambda x:round(x, 4))
        st.dataframe(df,use_container_width=True)

        # ★ Day18: Judgement 表示
        _render_judgement(judgement)
        ok = True
        # デバッグ用
        with st.expander("Raw result (debug)", expanded=False):
            st.json(result)
    finally:
        try:
            tmp_path.unlink()
        except FileNotFoundError:
            pass
    if ok:
        st.success("Day18完了：OK/NGと理由を表示できました。")
    else:
        st.error("Day18失敗")

if __name__ == "__main__":
    main()