"""Task 2: Hyperparameter Tuning"""
import pandas as pd
import numpy as np
import json
import os
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, make_scorer, mean_squared_error, root_mean_squared_error
import joblib

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "training_data.csv")
RESULTS_PATH = os.path.join(BASE_DIR, "results", "step2_s2.json")
MODELS_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Load data
df = pd.read_csv(DATA_PATH)
X = df.drop("turbine_output_mwh", axis=1)
y = df["turbine_output_mwh"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Parameter grid
param_grid = {
    "n_estimators": [50, 100, 200],
    "learning_rate": [0.05, 0.1, 0.2],
    "max_depth": [3, 5]
}

# Total combinations = 3*3*2 = 18
scorer = make_scorer(root_mean_squared_error, greater_is_better=False)

gb = GradientBoostingRegressor(random_state=42)
search = GridSearchCV(
    gb,
    param_grid=param_grid,
    cv=3,
    scoring=scorer,
    return_train_score=True
)

# Set experiment
mlflow.set_experiment("windcast-turbine-output-mwh")

# Parent run
with mlflow.start_run(run_name="tuning-windcast") as parent_run:
    mlflow.set_tag("priority", "high")

    search.fit(X_train, y_train)

    # Log each trial as nested run
    results_cv = search.cv_results_
    for i in range(len(results_cv["params"])):
        with mlflow.start_run(run_name=f"trial-{i+1}", nested=True):
            params = results_cv["params"][i]
            for k, v in params.items():
                mlflow.log_param(k, v)
            cv_rmse = -results_cv["mean_test_score"][i]
            mlflow.log_metric("cv_rmse", cv_rmse)

    # Best model
    best_model = search.best_estimator_
    best_params = search.best_params_
    best_cv_rmse = -search.best_score_

    # Evaluate on test set
    preds = best_model.predict(X_test)
    test_mae = mean_absolute_error(y_test, preds)
    test_rmse = np.sqrt(mean_squared_error(y_test, preds))

    mlflow.log_params(best_params)
    mlflow.log_metric("best_cv_rmse", best_cv_rmse)
    mlflow.log_metric("test_mae", test_mae)
    mlflow.log_metric("test_rmse", test_rmse)
    mlflow.sklearn.log_model(best_model, "best_tuned_model")

# Save best tuned model
joblib.dump(best_model, os.path.join(MODELS_DIR, "best_model.pkl"))

# Save results
result = {
    "search_type": "grid",
    "n_folds": 3,
    "total_trials": len(results_cv["params"]),
    "best_params": best_params,
    "best_mae": round(test_mae, 4),
    "best_cv_mae": round(best_cv_rmse, 4),
    "parent_run_name": "tuning-windcast"
}

with open(RESULTS_PATH, "w") as f:
    json.dump(result, f, indent=2)

print(f"Best params: {best_params}")
print(f"Best CV RMSE: {best_cv_rmse:.4f}")
print(f"Test MAE: {test_mae:.4f}")
print(f"Test RMSE: {test_rmse:.4f}")
print(f"Total trials: {len(results_cv['params'])}")
print(f"Results saved to {RESULTS_PATH}")
