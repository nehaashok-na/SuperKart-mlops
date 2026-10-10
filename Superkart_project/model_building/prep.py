from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_DIR / "data" / "sales.csv"
ARTIFACTS_DIR = PROJECT_DIR / "artifacts"
TARGET = "Product_Store_Sales_Total"
ID_COLUMNS = ["Product_Id", "Store_Id"]


def prepare_data() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH).drop_duplicates().copy()
    required = {TARGET}
    if not required.issubset(df.columns):
        raise ValueError(f"Target column {TARGET!r} is missing")

    if "Product_Sugar_Content" in df.columns:
        sugar = df["Product_Sugar_Content"].astype("string").str.strip()
        df["Product_Sugar_Content"] = sugar.replace(
            {"reg": "Regular", "REG": "Regular", "regular": "Regular"}
        )

    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df = df.dropna(subset=[TARGET])
    df = df[df[TARGET] >= 0].copy()
    if df.empty:
        raise ValueError("No valid rows remain after target cleaning.")

    X = df.drop(columns=[TARGET] + ID_COLUMNS, errors="ignore")
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    train_df = X_train.copy()
    train_df[TARGET] = y_train
    test_df = X_test.copy()
    test_df[TARGET] = y_test
    train_path = ARTIFACTS_DIR / "train.csv"
    test_path = ARTIFACTS_DIR / "test.csv"
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print("DATA PREPARATION PASSED")
    print(f"Rows used after cleaning: {len(df):,}")
    print(f"Training rows: {len(train_df):,}")
    print(f"Testing rows: {len(test_df):,}")
    print("Train split:", train_path)
    print("Test split:", test_path)


if __name__ == "__main__":
    prepare_data()
