# scripts/export_to_onnx.py
from __future__ import annotations
import os
import glob
import joblib

from sklearn.pipeline import Pipeline

from skl2onnx import convert_sklearn, update_registered_converter
from skl2onnx.common.data_types import FloatTensorType
from skl2onnx.common.shape_calculator import (
    calculate_linear_regressor_output_shapes,
)

from onnxmltools.convert.lightgbm.operator_converters.LightGbm import (
    convert_lightgbm,
)

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# ================== CONFIG ==================
ARTIFACT_DIR = "artifacts_113"
ONNX_DIR     = "onnx_models_113"

# Main ONNX opset và domain ai.onnx.ml opset
MAIN_OPSET      = 15
AI_ONNX_ML_OPSET = 3   # <— fix lỗi “domain 'ai.onnx.ml' version 5 not supported”
# ============================================


def register_external_converters() -> None:
    """
    Đăng ký converter cho:
      - LGBMRegressor (LightGBM)
    Dạng *regressor* để tránh route qua nhánh classifier.
    (XGBRegressor sẽ được SKIP, do env hiện tại hay lỗi khi convert.)
    """

    # ---- LGBMRegressor: bọc convert_lightgbm với option 'split' ----
    def skl2onnx_convert_lightgbm(scope, operator, container):
        # Lấy options, ví dụ options={"split": None}
        options = scope.get_options(operator.raw_operator)
        if "split" in options:
            operator.split = options["split"]
        else:
            operator.split = None
        convert_lightgbm(scope, operator, container)

    update_registered_converter(
        LGBMRegressor,
        "LightGbmLGBMRegressor",
        calculate_linear_regressor_output_shapes,
        skl2onnx_convert_lightgbm,
        options={"split": None},
    )

    # Ghi chú:
    # Với XGBRegressor, bạn đang gặp lỗi “could not convert string to float: '[2.8396597E1]'”
    # do tương thích version giữa xgboost và converter. Ở file này ta sẽ SKIP pipeline XGB,
    # thay vì cố convert trong env hiện tại.


def _is_xgb_pipeline(pipe) -> bool:
    """Kiểm tra bước cuối của pipeline có phải XGBRegressor không."""
    if not isinstance(pipe, Pipeline):
        return False
    last_step = pipe.steps[-1][1]
    return isinstance(last_step, XGBRegressor)


def export_pipeline_joblib_to_onnx(joblib_path: str, onnx_path: str) -> None:
    """
    Nạp 1 file joblib dạng:
        {"pipeline": pipe, "feature_cols": [...]}
    Convert toàn bộ pipeline sang ONNX.
    """
    print(f"\n--> Converting {os.path.basename(joblib_path)} -> {onnx_path}")

    obj = joblib.load(joblib_path)
    if not isinstance(obj, dict) or "pipeline" not in obj:
        raise ValueError(f"File {joblib_path} không phải định dạng artifact mong đợi.")

    pipe = obj["pipeline"]

    # Nếu là pipeline XGBRegressor => bỏ qua (env hiện tại đang lỗi khi convert)
    if _is_xgb_pipeline(pipe):
        print(
            "    [SKIP] Pipeline kết thúc bằng XGBRegressor – "
            "skl2onnx + xgboost trong env hiện tại đang lỗi khi convert.\n"
            "          Giữ nguyên .joblib cho horizon này, hoặc đổi env / đổi model nếu cần ONNX."
        )
        return

    # Lấy số feature đầu vào từ feature_cols
    feature_cols = obj.get("feature_cols", None)
    if feature_cols is None:
        raise ValueError(
            f"Artifact {joblib_path} không có 'feature_cols'; "
            "hãy đảm bảo scripts/tune.py đã lưu kèm key này."
        )

    n_features = len(feature_cols)

    # Định nghĩa kiểu input cho ONNX: batch x n_features, kiểu float
    initial_type = [("input", FloatTensorType([None, n_features]))]

    # target_opset: dict cho main domain "" và domain 'ai.onnx.ml'
    target_opset = {"": MAIN_OPSET, "ai.onnx.ml": AI_ONNX_ML_OPSET}

    # Convert pipeline sang ONNX
    onx = convert_sklearn(
        pipe,
        initial_types=initial_type,
        target_opset=target_opset,
    )

    # Tạo thư mục đích nếu chưa có
    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
    with open(onnx_path, "wb") as f:
        f.write(onx.SerializeToString())

    print(f"    [OK] Saved ONNX -> {onnx_path}")


def main():
    # 1) Đăng ký converter LightGBM (regressor)
    register_external_converters()

    # 2) Tìm tất cả pipeline_H*_*.joblib trong ARTIFACT_DIR
    pattern = os.path.join(ARTIFACT_DIR, "pipeline_H*_*.joblib")
    paths = sorted(glob.glob(pattern))
    if not paths:
        print(f"[WARN] Không tìm thấy file joblib nào trong pattern: {pattern}")
        return

    print(f"==> Found {len(paths)} joblib pipelines in '{ARTIFACT_DIR}'")

    # 3) Convert từng pipeline
    for path in paths:
        fname = os.path.basename(path)
        name_no_ext = os.path.splitext(fname)[0]  # vd: pipeline_H1_xgb
        onnx_path = os.path.join(ONNX_DIR, f"{name_no_ext}.onnx")

        try:
            export_pipeline_joblib_to_onnx(path, onnx_path)
        except Exception as e:
            print(f"    [ERROR] Failed to convert {fname}: {e}")


if __name__ == "__main__":
    main()
