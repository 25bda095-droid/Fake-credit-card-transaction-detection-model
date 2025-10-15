from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib
import io
from sklearn.metrics import roc_curve, auc, recall_score
import matplotlib.pyplot as plt
import base64

app = Flask(__name__)

MODELPATH = 'models/fraud_detection_model_tuned.pkl'
SCALERPATH = 'models/scaler.pkl'
RANDOMFORESTPATH = 'models/random_Forest_model.pkl'
XGBOOSTPATH = 'models/XGBoost_model.joblib'

# Load all models (cache in memory)
def load_models():
    tunedmodel = joblib.load(MODELPATH)
    scaler = joblib.load(SCALERPATH)
    rfmodel = joblib.load(RANDOMFORESTPATH)
    xgbmodel = joblib.load(XGBOOSTPATH)
    return tunedmodel, scaler, rfmodel, xgbmodel

tunedmodel, scaler, rfmodel, xgbmodel = load_models()

@app.route("/predict", methods=["POST"])
def predict():
    if 'file' not in request.files:
        return jsonify(success=False, error="No file uploaded"), 400
    file = request.files['file']
    filename = file.filename

    # Support CSV only in backend for simplicity
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
        'xgb': xgbmodel
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

# Example endpoint for ROC graph (returns PNG base64)
@app.route("/roc-graph", methods=["POST"])
def roc_graph():
    data = request.json
    y_true = np.array(data['y_true'])
    probs = {
        "tuned": np.array(data["tuned_probs"]),
        "rf": np.array(data["rf_probs"]),
        "xgb": np.array(data["xgb_probs"])
    }

    fig, ax = plt.subplots(figsize=(8, 6))
    for key, prob in probs.items():
        fpr, tpr, _ = roc_curve(y_true, prob)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{key.upper()} (AUC={roc_auc:.2f})")

    ax.plot([0, 1], [0, 1], color="gray", lw=2, linestyle="--")
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve Comparison')
    ax.legend(loc="lower right")
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode("utf8")
    plt.close(fig)
    return jsonify({"image": img_str})

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)
