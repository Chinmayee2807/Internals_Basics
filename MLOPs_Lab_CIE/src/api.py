"""Task 3: FastAPI Serving"""
import os
import json
import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel, Field

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.pkl")
RESULTS_PATH = os.path.join(BASE_DIR, "results", "step3_s4.json")

# Load model
model = joblib.load(MODEL_PATH)

app = FastAPI(title="WindCast Turbine Output API")


class TurbineInput(BaseModel):
    wind_speed_kmph: float = Field(..., ge=5, le=120)
    blade_length_m: float = Field(..., ge=20, le=80)
    altitude_m: float = Field(..., ge=50, le=200)
    humidity_pct: float = Field(..., ge=20, le=180)


class PredictionResponse(BaseModel):
    prediction: float


@app.get("/status")
def status():
    return {"status": "operational", "service": "WindCast API"}


@app.post("/infer")
def infer(data: TurbineInput):
    features = np.array([[data.wind_speed_kmph, data.blade_length_m, data.altitude_m, data.humidity_pct]])
    pred = model.predict(features)[0]
    return {"prediction": round(float(pred), 2)}


# --- Generate results JSON when run directly ---
if __name__ == "__main__":
    import requests
    import subprocess
    import time
    import sys

    # Start the server in background
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"],
        cwd=BASE_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    time.sleep(5)  # Wait for server to start

    try:
        # Test /status
        health_resp = requests.get("http://localhost:8000/status")
        health_json = health_resp.json()
        print(f"Health: {health_json}")

        # Test /infer
        test_input = {"wind_speed_kmph": 35.4, "blade_length_m": 42.4, "altitude_m": 125.3, "humidity_pct": 43.1}
        pred_resp = requests.post("http://localhost:8000/infer", json=test_input)
        pred_json = pred_resp.json()
        print(f"Prediction: {pred_json}")

        # Save results
        result = {
            "health_endpoint": "/status",
            "predict_endpoint": "/infer",
            "port": 8000,
            "health_response": health_json,
            "test_input": test_input,
            "prediction": pred_json["prediction"]
        }

        os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
        with open(RESULTS_PATH, "w") as f:
            json.dump(result, f, indent=2)

        print(f"Results saved to {RESULTS_PATH}")

    finally:
        proc.terminate()
        proc.wait()
