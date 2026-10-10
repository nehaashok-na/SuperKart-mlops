from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_DIR / "data" / "sales.csv"
TARGET = "Product_Store_Sales_Total"
EXPECTED_COLUMNS = [
    "Product_Id", "Product_Weight", "Product_Sugar_Content",
    "Product_Allocated_Area", "Product_Type", "Product_MRP",
    "Store_Id", "Store_Establishment_Year", "Store_Size",
    "Store_Location_City_Type", "Store_Type", TARGET,
]


def validate_dataset() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    missing_columns = sorted(set(EXPECTED_COLUMNS) - set(df.columns))
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {missing_columns}")

    if df.empty:
        raise ValueError("Dataset is empty.")

    print("DATA REGISTRATION / VALIDATION PASSED")
    print(f"Rows: {df.shape[0]:,}")
    print(f"Columns: {df.shape[1]:,}")
    print("Column names:", df.columns.tolist())
    print("\nMissing values by column:\n", df[EXPECTED_COLUMNS].isna().sum().to_string())
    print(f"\nDuplicate rows: {df.duplicated().sum():,}")

    target = pd.to_numeric(df[TARGET], errors="coerce")
    print("\nTarget summary:")
    print(target.describe().to_string())
    print("\nCategorical value examples:")
    for column in ["Product_Sugar_Content", "Product_Type", "Store_Size", "Store_Location_City_Type", "Store_Type"]:
        print(f"- {column}: {df[column].dropna().astype(str).unique().tolist()}")
    return df


if __name__ == "__main__":
    validate_dataset()
