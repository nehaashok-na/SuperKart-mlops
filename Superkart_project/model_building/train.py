from pathlib import Path
import json
import joblib
import mlflow
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

PROJECT_DIR = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = PROJECT_DIR / "artifacts"
DEPLOYMENT_DIR = PROJECT_DIR / "deployment"
TARGET = "Product_Store_Sales_Total"


def train_model() -> dict:
    train_path = ARTIFACTS_DIR / "train.csv"
    test_path = ARTIFACTS_DIR / "test.csv"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError("Train/test splits are missing. Run prep.py first.")
    DEPLOYMENT_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    X_train, y_train = train.drop(columns=[TARGET]), train[TARGET]
    X_test, y_test = test.drop(columns=[TARGET]), test[TARGET]

    numeric_cols = X_train.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = X_train.select_dtypes(exclude=["number"]).columns.tolist()

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_cols),
        ("categorical", categorical_pipeline, categorical_cols),
    ])
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", RandomForestRegressor(random_state=42, n_jobs=1)),
    ])

    # Four candidate configurations keep the first end-to-end run practical.
    param_grid = {
        "model__n_estimators": [25, 50],
        "model__max_depth": [10, 15],
    }
    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="neg_mean_absolute_error",
        cv=2,
        n_jobs=1,
        verbose=1,
        refit=True,
    )

    tracking_uri = (PROJECT_DIR / "mlruns").resolve().as_uri()
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("sales-revenue-prediction")

    model_path = DEPLOYMENT_DIR / "best_model.joblib"
    metrics_path = ARTIFACTS_DIR / "metrics.json"
    cv_path = ARTIFACTS_DIR / "cv_results.csv"

    with mlflow.start_run() as run:
        search.fit(X_train, y_train)
        best_model = search.best_estimator_
        predictions = best_model.predict(X_test)
        metrics = {
            "mae": float(mean_absolute_error(y_test, predictions)),
            "rmse": float(mean_squared_error(y_test, predictions) ** 0.5),
            "r2": float(r2_score(y_test, predictions)),
        }
        joblib.dump(best_model, model_path)
        pd.DataFrame(search.cv_results_).to_csv(cv_path, index=False)
        with metrics_path.open("w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2)

        mlflow.log_param("algorithm", "RandomForestRegressor")
        mlflow.log_param("cv_folds", 2)
        mlflow.log_param("target", TARGET)
        mlflow.log_params({key: str(value) for key, value in search.best_params_.items()})
        mlflow.log_metrics(metrics)
        mlflow.set_tag("selection_metric", "cross-validation MAE")
        mlflow.log_artifact(str(model_path))
        mlflow.log_artifact(str(metrics_path))
        mlflow.log_artifact(str(cv_path))

        print("BEST PARAMETERS:", search.best_params_)
        print("HELD-OUT TEST METRICS:", json.dumps(metrics, indent=2))
        print("MODEL SAVED:", model_path)
        print("MLFLOW RUN ID:", run.info.run_id)
        return metrics


if __name__ == "__main__":
    train_model()
