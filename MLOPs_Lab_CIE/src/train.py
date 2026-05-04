"""Task 1: Experiment Tracking & Model Comparison"""
import pandas as pd
import numpy as np
import json
import os
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
import joblib

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "training_data.csv")
RESULTS_PATH = os.path.join(BASE_DIR, "results", "step1_s1.json")
MODELS_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Load data
df = pd.read_csv(DATA_PATH)
X = df.drop("turbine_output_mwh", axis=1)
y = df["turbine_output_mwh"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Set experiment
mlflow.set_experiment("windcast-turbine-output-mwh")

models_info = []

# --- Train Lasso ---
with mlflow.start_run(run_name="Lasso"):
    mlflow.set_tag("priority", "high")
    lasso = Lasso(alpha=1.0, random_state=42)
    lasso.fit(X_train, y_train)
    preds = lasso.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    mape = mean_absolute_percentage_error(y_test, preds)
    mlflow.log_param("alpha", 1.0)
    mlflow.log_param("random_state", 42)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2)
    mlflow.log_metric("mape", mape)
    mlflow.sklearn.log_model(lasso, "model")
    models_info.append({"name": "Lasso", "mae": round(mae, 4), "rmse": round(rmse, 4), "r2": round(r2, 4), "mape": round(mape, 4)})
    print(f"Lasso -> MAE: {mae:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}, MAPE: {mape:.4f}")

# --- Train GradientBoosting ---
with mlflow.start_run(run_name="GradientBoosting"):
    mlflow.set_tag("priority", "high")
    gb = GradientBoostingRegressor(n_estimators=100, random_state=42)
    gb.fit(X_train, y_train)
    preds = gb.predict(X_test)
    mae_gb = mean_absolute_error(y_test, preds)
    rmse_gb = np.sqrt(mean_squared_error(y_test, preds))
    r2_gb = r2_score(y_test, preds)
    mape_gb = mean_absolute_percentage_error(y_test, preds)
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("random_state", 42)
    mlflow.log_metric("mae", mae_gb)
    mlflow.log_metric("rmse", rmse_gb)
    mlflow.log_metric("r2", r2_gb)
    mlflow.log_metric("mape", mape_gb)
    mlflow.sklearn.log_model(gb, "model")
    models_info.append({"name": "GradientBoosting", "mae": round(mae_gb, 4), "rmse": round(rmse_gb, 4), "r2": round(r2_gb, 4), "mape": round(mape_gb, 4)})
    print(f"GradientBoosting -> MAE: {mae_gb:.4f}, RMSE: {rmse_gb:.4f}, R2: {r2_gb:.4f}, MAPE: {mape_gb:.4f}")

# Select best by RMSE
best = min(models_info, key=lambda x: x["rmse"])

# Save best model
if best["name"] == "Lasso":
    joblib.dump(lasso, os.path.join(MODELS_DIR, "best_model.pkl"))
else:
    joblib.dump(gb, os.path.join(MODELS_DIR, "best_model.pkl"))

# Save results JSON
result = {
    "experiment_name": "windcast-turbine-output-mwh",
    "models": models_info,
    "best_model": best["name"],
    "best_metric_name": "rmse",
    "best_metric_value": best["rmse"]
}

with open(RESULTS_PATH, "w") as f:
    json.dump(result, f, indent=2)

print(f"\nBest model: {best['name']} with RMSE: {best['rmse']}")
print(f"Results saved to {RESULTS_PATH}")
