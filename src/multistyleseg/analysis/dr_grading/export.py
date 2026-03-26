import json
import numpy as np


def export_pipeline(
    pipeline,
    feature_names: list[str],
    path: str = "logistic_model.json",
    modelname="logisticregression",
    threshold: float = 0.5,
):
    scaler = pipeline.named_steps["standardscaler"]
    lr = pipeline.named_steps[modelname]

    payload = {
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "coef": lr.coef_[0].tolist() if lr.coef_.ndim == 2 else lr.coef_.tolist(),
        "intercept": float(lr.intercept_[0])
        if isinstance(lr.intercept_, np.ndarray)
        else float(lr.intercept_),
        "threshold": threshold,
        "feature_names": feature_names,
    }

    with open(path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"Exported model to {path}")
    print(f"  {len(feature_names)} features")
    print(f"  intercept = {payload['intercept']:.6f}")
    print(f"  threshold = {threshold}")

    # ── sanity check: compare Python prediction vs manual computation ──
    if hasattr(pipeline, "predict_proba"):
        # Use a dummy input to verify round-trip
        dummy_x = np.zeros((1, len(feature_names)))
        p_sklearn = pipeline.predict_proba(dummy_x)[0, 1]

        z = (dummy_x[0] - scaler.mean_) / scaler.scale_
        logit = z @ lr.coef_[0] + lr.intercept_[0]
        p_manual = 1.0 / (1.0 + np.exp(-logit))

        print(
            f"  sanity check (zero input): sklearn={p_sklearn:.8f}  manual={p_manual:.8f}  match={np.isclose(p_sklearn, p_manual)}"
        )
