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
import json
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))
    
from bioimage_qc.pipeline import evaluate_image
from bioimage_qc.exceptions import ImageQualityError

ALLOWED_EXTS = {".jpg", ".jpeg", ".png"}

def _validate_uploaded_file(uploaded_file) -> None:
    """アップロード直後の基本入力チェック。"""
    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix not in ALLOWED_EXTS:
        raise ImageQualityError(f"未対応の拡張子です: {suffix}")

    data = uploaded_file.getvalue()
    if not data:
        raise ImageQualityError("アップロードファイルが空です。画像を再選択してください。")
    
def _decode_image(uploaded_file)-> np.ndarray:
    """StreamlitのUploadedFile（バイト列）をOpenCV画像（BGR）へデコードする"""
    data = uploaded_file.getvalue()
    arr = np.frombuffer(data, dtype=np.uint8)
    img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ImageQualityError("画像のデコードに失敗しました（破損/形式不正の可能性があります）")
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

def _build_csv_row(uploaded_name: str, result: dict, thresholds: dict) -> dict:
    """CSV出力用に1件分の結果を整形する。"""
    metrics = result.get("metrics", {})
    judgement = result.get("judgement", {})
    reasons = judgement.get("reasons", [])
    reasons_str = "; ".join(map(str, reasons)) if isinstance(reasons, list) else str(reasons)
    return {
        "evaluated_at": datetime.now().isoformat(timespec="seconds"),
        "file_name": uploaded_name,
        "input_path": str(result.get("input_path", "")),
        "ok": bool(judgement.get("ok", False)),
        "label": str(judgement.get("label", "OK" if judgement.get("ok", False) else "NG")),
        "reasons": reasons_str,
        "brightness_mean": float(metrics.get("brightness_mean", float("nan"))),
        "contrast_std": float(metrics.get("contrast_std", float("nan"))),
        "sharpness_lap_var": float(metrics.get("sharpness_lap_var", float("nan"))),
        "thresholds_json": json.dumps(thresholds, ensure_ascii=False),
    }

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

    # Evaluate (preview + metrics + judgement)
    tmp_path: Path | None = None
    ok = False
    try:
        _validate_uploaded_file(uploaded)

        # Preview
        img_bgr = _decode_image(uploaded)
        if img_bgr.size == 0:
            raise ImageQualityError("画像データが空です。")
        if img_bgr.ndim != 3 or img_bgr.shape[2] != 3:
            raise ImageQualityError(f"想定外の画像形状です: shape={img_bgr.shape}")
        h, w = img_bgr.shape[:2]
        if h <= 0 or w <= 0:
            raise ImageQualityError(f"画像サイズが不正です: width={w}, height={h}")

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        st.subheader("Preview")
        st.image(img_rgb, caption=uploaded.name, use_container_width=True)

        tmp_path = _save_to_temp(uploaded)
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

        # Day19: CSVダウンロード（1件結果）
        st.subheader("Export CSV")
        csv_row = _build_csv_row(uploaded.name, result, thresholds)
        export_df = pd.DataFrame([csv_row])
        csv_bytes = export_df.to_csv(index=False).encode("utf-8-sig")
        out_name = f"qc_result_{Path(uploaded.name).stem}.csv"
        st.download_button(
            label="結果をCSVでダウンロード",
            data=csv_bytes,
            file_name=out_name,
            mime="text/csv",
        )

        ok = True
        # デバッグ用
        with st.expander("Raw result (debug)", expanded=False):
            st.json(result)
    except ImageQualityError as e:
        st.error(f"入力エラー: {e}")
    except ValueError as e:
        st.error(f"入力値エラー: {e}")
    except Exception as e:
        st.error("予期しないエラーが発生しました。入力画像や設定値を確認してください。")
        with st.expander("Error detail (debug)", expanded=False):
            st.exception(e)
    finally:
        if tmp_path is not None:
            try:
                tmp_path.unlink()
            except FileNotFoundError:
                pass

    if ok:
        st.success("Day18完了：OK/NGと理由を表示できました。")

if __name__ == "__main__":
    main()
