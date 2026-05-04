"""Task 4: Retraining Pipeline"""
import pandas as pd
import numpy as np
import json
import os
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import joblib

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_DATA = os.path.join(BASE_DIR, "data", "training_data.csv")
NEW_DATA = os.path.join(BASE_DIR, "data", "new_data.csv")
RESULTS_PATH = os.path.join(BASE_DIR, "results", "step4_s8.json")
MODELS_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Load step2 results to get best params
STEP2_PATH = os.path.join(BASE_DIR, "results", "step2_s2.json")
with open(STEP2_PATH, "r") as f:
    step2 = json.load(f)
best_params = step2["best_params"]

# Load data
df_orig = pd.read_csv(TRAIN_DATA)
df_new = pd.read_csv(NEW_DATA)

original_rows = len(df_orig)
new_rows = len(df_new)

# Combined data
df_combined = pd.concat([df_orig, df_new], ignore_index=True)
combined_rows = len(df_combined)

X_combined = df_combined.drop("turbine_output_mwh", axis=1)
y_combined = df_combined["turbine_output_mwh"]

# Split combined data for evaluation
X_train_comb, X_test_comb, y_train_comb, y_test_comb = train_test_split(
    X_combined, y_combined, test_size=0.2, random_state=42
)

# Champion: load the existing best model
champion = joblib.load(os.path.join(MODELS_DIR, "best_model.pkl"))
champion_preds = champion.predict(X_test_comb)
champion_rmse = np.sqrt(mean_squared_error(y_test_comb, champion_preds))

# --- Retrained model: train on combined training set ---
mlflow.set_experiment("windcast-turbine-output-mwh")

with mlflow.start_run(run_name="retrain-windcast"):
    mlflow.set_tag("priority", "high")

    retrained = GradientBoostingRegressor(random_state=42, **best_params)
    retrained.fit(X_train_comb, y_train_comb)

    retrained_preds = retrained.predict(X_test_comb)
    retrained_rmse = np.sqrt(mean_squared_error(y_test_comb, retrained_preds))

    mlflow.log_params(best_params)
    mlflow.log_metric("champion_rmse", champion_rmse)
    mlflow.log_metric("retrained_rmse", retrained_rmse)
    mlflow.sklearn.log_model(retrained, "retrained_model")

improvement = champion_rmse - retrained_rmse
threshold = 0.3

if improvement >= threshold:
    action = "promoted"
    joblib.dump(retrained, os.path.join(MODELS_DIR, "best_model.pkl"))
    print("Retrained model PROMOTED.")
else:
    action = "kept_champion"
    print("Champion model KEPT.")

result = {
    "original_data_rows": original_rows,
    "new_data_rows": new_rows,
    "combined_data_rows": combined_rows,
    "champion_rmse": round(champion_rmse, 4),
    "retrained_rmse": round(retrained_rmse, 4),
    "improvement": round(improvement, 4),
    "min_improvement_threshold": threshold,
    "action": action,
    "comparison_metric": "rmse"
}

with open(RESULTS_PATH, "w") as f:
    json.dump(result, f, indent=2)

print(f"Champion RMSE: {champion_rmse:.4f}")
print(f"Retrained RMSE: {retrained_rmse:.4f}")
print(f"Improvement: {improvement:.4f}")
print(f"Action: {action}")
print(f"Results saved to {RESULTS_PATH}")
