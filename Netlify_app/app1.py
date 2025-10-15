from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import roc_curve, auc, recall_score

app = Flask(__name__)

# ... load models and define other routes here ...

@app.route("/")
def home():
    return "Fraud Detection API is running."

@app.route("/predict", methods=["POST"])
def predict():


MODELPATH = 'models/fraud_detection_model_tuned.pkl'
SCALERPATH = 'models/scaler.pkl'
RANDOMFORESTPATH = 'models/random_Forest_model.pkl'
LOGISTICPATH = 'models/LogisticRegression_model.joblib'

# Load all models (cache in memory)
def load_models():
    tunedmodel = joblib.load(MODELPATH)
    scaler = joblib.load(SCALERPATH)
    rfmodel = joblib.load(RANDOMFORESTPATH)
    logitmodel = joblib.load(LOGISTICPATH)
    return tunedmodel, scaler, rfmodel, logitmodel

tunedmodel, scaler, rfmodel, logitmodel = load_models()

@app.route("/predict", methods=["POST"])
def predict():
    if 'file' not in request.files:
        return jsonify(success=False, error="No file uploaded"), 400
    file = request.files['file']

    try:
        df = pd.read_csv(file)
    except Exception as e:
        return jsonify(success=False, error=f"Upload error: {e}"), 400

    required_cols = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        if 'Time' in missing_cols:
            required_cols = [f'V{i}' for i in range(1, 29)] + ['Amount']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            return jsonify(success=False, error=f"Missing columns: {', '.join(missing_cols)}"), 400

    for col in required_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=required_cols)
    if len(df) == 0:
        return jsonify(success=False, error="No valid data found after cleaning"), 400

    X = df[required_cols].copy()
    X_scaled = scaler.transform(X)

    # Collect predictions from all models
    models_choices = {
        'tuned': tunedmodel,
        'rf': rfmodel,
        'logistic': logitmodel
    }
    results = {}
    for key, model in models_choices.items():
        pred = model.predict(X_scaled)
        prob = model.predict_proba(X_scaled)[:, 1]
        results[key] = {'pred': pred.tolist(), 'prob': prob.tolist()}

    # If true labels available
    y_true = df['Class'].values.tolist() if 'Class' in df.columns else None

    response = {
        'success': True,
        'results': results,
        'total_transactions': len(X_scaled),
        'y_true': y_true
    }
    return jsonify(response)

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)
