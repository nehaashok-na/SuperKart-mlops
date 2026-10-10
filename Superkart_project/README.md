# Superkart MLOps Pipeline

## Overview

This repository contains an automated machine-learning pipeline for predicting product-store sales revenue from product and store attributes. It includes dataset validation, data preparation, Random Forest hyperparameter tuning, MLflow experiment tracking, model evaluation, GitHub Actions CI/CD, and a Streamlit prediction app.

> Scope note: the supplied CSV does not contain sales dates or future sales-pipeline fields. The current model predicts product-store revenue; it is not a true weekly, monthly, quarterly, or annual time-series forecast.

## Technology Stack

Python, pandas, scikit-learn, MLflow, joblib, GitHub Actions, and Streamlit Community Cloud.

## Repository Structure

```text
sSuperkart_project-mlops/
├── .github/workflows/pipeline.yml
├── Superkart_project/
│   ├── data/sales.csv
│   ├── model_building/
│   │   ├── data_register.py
│   │   ├── prep.py
│   │   ├── train.py
│   │   └── evaluate.py
│   ├── deployment/
│   │   ├── app.py
│   │   ├── requirements.txt
│   │   └── best_model.joblib  # created by the successful workflow
│   ├── requirements.txt
│   └── README.md
└── README.md
```

## Pipeline

1. Validate the required CSV schema and report data quality.
2. Clean the data and generate reproducible 80/20 train/test splits.
3. Tune a Random Forest regressor using cross-validation MAE.
4. Log parameters, metrics, the candidate CV results, and model artifact with MLflow.
5. Evaluate the selected model on the held-out test set using MAE, RMSE, and R-squared.
6. Run validation, preparation, training, and evaluation through GitHub Actions; pass splits between jobs as workflow artifacts.
7. Commit the trained model to the repository on successful `main`-branch runs.
8. Deploy `Superkart_project/deployment/app.py` through Streamlit Community Cloud.

## Run Locally

Install `Superkart_project/requirements.txt`, then run:

```bash
python Superkart_project/model_building/data_register.py
python Superkart_project/model_building/prep.py
python Superkart_project/model_building/train.py
python Superkart_project/model_building/evaluate.py
```

## Evaluation

After the successful pipeline run, add the actual held-out test metrics (MAE, RMSE, R-squared) here. Do not invent performance values.

## Deployment

- GitHub repository: add/confirm this repository URL.
- Streamlit application: add your deployed `https://...streamlit.app` URL.
- GitHub Actions: link to a successful workflow run.
