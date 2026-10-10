from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

PROJECT_DIR = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = PROJECT_DIR / "artifacts"
MODEL_PATH = PROJECT_DIR / "deployment" / "best_model.joblib"
TARGET = "Product_Store_Sales_Total"


def evaluate_model() -> dict:
    test_path = ARTIFACTS_DIR / "test.csv"
    if not test_path.exists():
        raise FileNotFoundError(f"Test split not found: {test_path}")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Saved model not found: {MODEL_PATH}")

    test = pd.read_csv(test_path)
    X_test = test.drop(columns=[TARGET])
    y_test = test[TARGET]
    model = joblib.load(MODEL_PATH)
    predictions = model.predict(X_test)

    metrics = {
        "mae": float(mean_absolute_error(y_test, predictions)),
        "rmse": float(mean_squared_error(y_test, predictions) ** 0.5),
        "r2": float(r2_score(y_test, predictions)),
    }
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS_DIR / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    pd.DataFrame({"actual_sales": y_test, "predicted_sales": predictions}).to_csv(
        ARTIFACTS_DIR / "predictions.csv", index=False
    )
    print("INDEPENDENT TEST EVALUATION")
    print(json.dumps(metrics, indent=2))
    print("Metrics file:", ARTIFACTS_DIR / "metrics.json")
    print("Prediction file:", ARTIFACTS_DIR / "predictions.csv")
    return metrics


if __name__ == "__main__":
    evaluate_model()
