import streamlit as st
import pandas as pd
import numpy as np
import joblib
import PyPDF2
import re
import io
import os

# ---- LOAD MODELS AND SCALER ----
MODELPATH = 'fraud_detection_model_tuned.pkl'
SCALERPATH = 'scaler.pkl'
RANDOMFORESTPATH = 'random_Forest_model.pkl'


@st.cache_resource
def load_models():
    try:
        tunedmodel = joblib.load(MODELPATH)
        scaler = joblib.load(SCALERPATH)
        rfmodel = joblib.load(RANDOMFORESTPATH)
        return tunedmodel, scaler, rfmodel
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None

tunedmodel, scaler, rfmodel = load_models()

ALLOWED_EXTENSIONS = ['csv', 'pdf']

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_csv_from_pdf(pdf_file):
    try:
        pdfreader = PyPDF2.PdfReader(pdf_file)
        text = ''
        for page in pdfreader.pages:
            text += page.extract_text()
        lines = text.strip().split('\n')
        data = []
        for line in lines:
            row = re.split(r',\s*', line.strip())
            if len(row) > 1:
                data.append(row)
        if data:
            df = pd.DataFrame(data[1:], columns=data[0])
            return df
        else:
            return None
    except Exception as e:
        st.error(f"PDF extraction error: {e}")
        return None

def preprocess_data(df):
    try:
        required_cols = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            if 'Time' in missing_cols:
                required_cols = [f'V{i}' for i in range(1, 29)] + ['Amount']
                missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                return None, f"Missing required columns: {', '.join(missing_cols)}"
        for col in required_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.dropna(subset=required_cols)
        if len(df) == 0:
            return None, "No valid data found after cleaning."
        feature_cols = required_cols
        X = df[feature_cols].copy()
        X_scaled = scaler.transform(X)
        return X_scaled, None
    except Exception as e:
        return None, f"Preprocessing error: {str(e)}"

def predict_fraud(X_scaled, model_choice):
    try:
        model = tunedmodel if model_choice == "tuned" else rfmodel
        predictions = model.predict(X_scaled)
        probabilities = model.predict_proba(X_scaled)[:, 1]
        return predictions, probabilities
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None, None

# ---- STREAMLIT APP UI ----
st.title("Fraud Detection System")
st.write("AI-Powered Credit Card Transaction Analysis")

st.info("""
**Instructions**
- Upload your credit card transaction data in CSV or PDF format.
- The file must contain columns: Time, V1-V28, Amount.
- Select your preferred ML model for analysis.
- Get instant fraud detection results!
""")

uploaded_file = st.file_uploader("Choose your file (CSV or PDF)", type=ALLOWED_EXTENSIONS)
model_choice = st.radio(
    "Select Model",
    [("Tuned Model", "tuned"), ("Random Forest Model", "rf")],
    format_func=lambda x: x[0]
)[1]
analyze = st.button("Analyze Transactions")

if analyze and uploaded_file is not None:
    filename = uploaded_file.name
    if not allowed_file(filename):
        st.error("Invalid file format. Please upload a CSV or PDF.")
    else:
        try:
            if filename.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = extract_csv_from_pdf(uploaded_file)
            if df is None:
                st.error("Could not extract data from the uploaded file.")
            else:
                X_scaled, error = preprocess_data(df)
                if error:
                    st.error(error)
                else:
                    predictions, probabilities = predict_fraud(X_scaled, model_choice)
                    if predictions is None:
                        st.error("Prediction failed.")
                    else:
                        total_transactions = len(predictions)
                        fraud_count = int(np.sum(predictions))
                        legitimate_count = total_transactions - fraud_count
                        fraud_percentage = (fraud_count / total_transactions) * 100
                        st.success(f"Total Transactions: {total_transactions}")
                        st.warning(f"Fraudulent: {fraud_count} ({fraud_percentage:.2f}%)")
                        st.info(f"Legitimate: {legitimate_count}")
                        # High Risk Transactions
                        st.subheader("High-Risk Transactions (Prob > 0.70)")
                        high_risk_indices = np.where(probabilities > 0.70)[0]
                        if len(high_risk_indices) == 0:
                            st.write("No high-risk transactions detected!")
                        else:
                            for idx in high_risk_indices[:10]:  # Top 10
                                st.write(f"Transaction #{idx+1} - Probability: {probabilities[idx]*100:.1f}%, Amount: {df.iloc[idx]['Amount'] if 'Amount' in df.columns else 'N/A'}")

        except Exception as e:
            st.error(f"Processing error: {str(e)}")
