"""
Task 05 - ML Prediction API (Flask)
Exposes:
 - /           → Service info
 - /health     → Health check
 - /predict    → Predict using trained ML model
"""

from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)


@app.route("/")
def home():
    return {"service": "Task 05 ML App", "status": "running"}


@app.route("/health")
def health():
    return {"health": "ok", "message": "Service is healthy"}


@app.route("/predict", methods=["POST"])
def predict():
    """
    Expected JSON:
    {
        "features": [5.1, 3.5, 1.4, 0.2]
    }
    """

    data = request.get_json()

    if not data or "features" not in data:
        return jsonify({"error": "Missing 'features' field"}), 400

    try:
        features = np.array(data["features"]).reshape(1, -1)
        prediction = int(model.predict(features)[0])
        return {"prediction": prediction, "success": True}

    except Exception as e:
        return jsonify({"error": str(e), "success": False}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
