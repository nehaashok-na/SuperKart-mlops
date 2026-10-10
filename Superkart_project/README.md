# Superkart MLOps Project

This project predicts `Product_Store_Sales_Total` from product and store attributes using a tuned Random Forest regression pipeline. It automates validation, preparation, experiment tracking, evaluation, model artifact management, and Streamlit deployment.

## Folders

- `data/`: source `sales.csv`
- `model_building/`: validation, preparation, training, and evaluation scripts
- `artifacts/`: train/test splits, metrics, CV results, and predictions generated during a run
- `deployment/`: Streamlit app, deployment requirements, and trained model
- `mlruns/`: local MLflow tracking data (do not commit this folder)

## Local execution

```bash
python Superkart_project/model_building/data_register.py
python Superkart_project/model_building/prep.py
python Superkart_project/model_building/train.py
python Superkart_project/model_building/evaluate.py
```

The source dataset has no date field; the current task is product-store revenue prediction, not a genuine time-series forecast.
