from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

PROJECT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_DIR / "deployment" / "best_model.joblib"

st.set_page_config(page_title="Sales Revenue Predictor", page_icon="📈", layout="centered")
st.title("Product-Store Sales Revenue Prediction")
st.write(
    "Enter product and store characteristics to estimate product-store revenue. "
    "This is a record-level prediction, not a date-based weekly/monthly forecast."
)

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found at {MODEL_PATH}. Check that the GitHub Actions pipeline committed it."
        )
    return joblib.load(MODEL_PATH)

try:
    model = load_model()
except Exception as exc:
    st.error(f"Unable to load the trained model: {exc}")
    st.stop()

with st.form("sales_prediction_form"):
    st.subheader("Product details")
    product_weight = st.number_input("Product weight", min_value=0.0, value=12.0, step=0.1)
    sugar = st.selectbox("Product sugar content", ["Low Sugar", "Regular", "No Sugar"])
    allocated_area = st.number_input("Product allocated display area", min_value=0.0, value=0.05, step=0.001, format="%.3f")
    product_type = st.selectbox(
        "Product type",
        [
            "Baking Goods", "Breads", "Breakfast", "Canned", "Dairy",
            "Frozen Foods", "Fruits and Vegetables", "Hard Drinks",
            "Health and Hygiene", "Household", "Meat", "Others",
            "Seafood", "Snack Foods", "Soft Drinks", "Starchy Foods",
        ],
    )
    mrp = st.number_input("Product MRP", min_value=0.0, value=150.0, step=1.0)

    st.subheader("Store details")
    establishment_year = st.number_input(
        "Store establishment year", min_value=1980, max_value=2026, value=2000, step=1
    )
    store_size = st.selectbox("Store size", ["Small", "Medium", "High"])
    city_type = st.selectbox("Store location city type", ["Tier 1", "Tier 2", "Tier 3"])
    store_type = st.selectbox(
        "Store type",
        ["Departmental Store", "Food Mart", "Supermarket Type1", "Supermarket Type2"],
    )
    submitted = st.form_submit_button("Predict sales revenue", type="primary")

if submitted:
    input_df = pd.DataFrame([{
        "Product_Weight": product_weight,
        "Product_Sugar_Content": sugar,
        "Product_Allocated_Area": allocated_area,
        "Product_Type": product_type,
        "Product_MRP": mrp,
        "Store_Establishment_Year": int(establishment_year),
        "Store_Size": store_size,
        "Store_Location_City_Type": city_type,
        "Store_Type": store_type,
    }])
    try:
        prediction = float(model.predict(input_df)[0])
        st.success("Prediction completed")
        st.metric("Predicted product-store sales revenue", f"{prediction:,.2f}")
        with st.expander("View submitted inputs"):
            st.dataframe(input_df, use_container_width=True)
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
