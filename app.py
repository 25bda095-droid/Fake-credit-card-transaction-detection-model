import streamlit as st
import pandas as pd
import numpy as np
import joblib
import PyPDF2
import re
import io
import os
from PIL import Image


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

st.markdown("""
<div style='display:flex;flex-direction:column;align-items:center;justify-content:center;margin-top:2em;'>
  <img src='https://cdn-icons-png.flaticon.com/512/3135/3135715.png' width='90' style='margin-bottom:1em;border-radius:50%;box-shadow:0 2px 12px #ffb80055;'>
  <h1 style='margin-bottom:0.2em;font-size:2.8rem;font-weight:800;letter-spacing:1px;'>Fraud Detection System</h1>
  <h4 style='color:#ffb800;font-size:1.3rem;font-weight:600;margin-bottom:1.2em;'>CREDIT CARD TRANSACTION ANALYSIS</h4>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div style='background:#232946;border-radius:18px;padding:22px 32px;margin-bottom:22px;color:#f4f4f4;box-shadow:0 4px 24px rgba(0,0,0,0.13);max-width:600px;margin-left:auto;margin-right:auto;'>
<b style='font-size:1.1em;color:#ffb800;'>How to Use:</b><br><br>
<ul style='font-size:1.08em;line-height:1.7;'>
<li>Upload your credit card transaction data (<b>CSV</b> or <b>PDF</b>).</li>
<li>File must have columns: <b>Time, V1-V28, Amount</b>.</li>
<li>Select your preferred ML model for analysis.</li>
<li>Click <b>Analyze Transactions</b> to get instant fraud detection results!</li>
</ul>
<span style='color:#ff4b4b;'>Your data is processed securely and never stored.</span>
</div>
""", unsafe_allow_html=True)

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
                        st.markdown(f"<div style='text-align:center;margin-bottom:18px;'><b>Total Transactions:</b> {total_transactions}</div>", unsafe_allow_html=True)
                        st.markdown(f"<div style='color:#fc5c7d;font-weight:600;'>Fraudulent: {fraud_count} ({fraud_percentage:.2f}%)</div>", unsafe_allow_html=True)
                        st.markdown(f"<div style='color:#6a82fb;font-weight:600;'>Legitimate: {legitimate_count}</div>", unsafe_allow_html=True)
                        # High Risk Transactions
                        st.markdown("<hr style='margin:18px 0;'>", unsafe_allow_html=True)
                        st.markdown("<div style='font-weight:600;font-size:1.1rem;margin-bottom:8px;'>High-Risk Transactions (Prob > 0.70)</div>", unsafe_allow_html=True)

                        # --- Charts Section ---
                        import matplotlib.pyplot as plt
                        import seaborn as sns
                        import pandas as pd
                        # Pie chart for fraud vs legit
                        fig1, ax1 = plt.subplots(figsize=(3,3))
                        ax1.pie([fraud_count, legitimate_count], labels=['Fraudulent', 'Legitimate'], autopct='%1.1f%%', colors=['#fc5c7d', '#6a82fb'], startangle=90, textprops={'color':'#232946','fontsize':12})
                        ax1.set_title('Fraud vs Legitimate', fontsize=14, color='#232946')
                        st.pyplot(fig1)

                        # Bar chart for probability distribution
                        fig2, ax2 = plt.subplots(figsize=(5,2.5))
                        sns.histplot(probabilities, bins=20, kde=True, color='#6a82fb', ax=ax2)
                        ax2.set_xlabel('Fraud Probability')
                        ax2.set_ylabel('Count')
                        ax2.set_title('Probability Distribution of Transactions', color='#232946')
                        st.pyplot(fig2)

                        # If you want a table of high-risk transactions:
                        high_risk_indices = np.where(probabilities > 0.70)[0]
                        if len(high_risk_indices) > 0:
                            st.markdown('<b>High-Risk Transactions Table</b>', unsafe_allow_html=True)
                            high_risk_df = df.iloc[high_risk_indices].copy()
                            high_risk_df['Fraud Probability'] = probabilities[high_risk_indices]
                            st.dataframe(high_risk_df[['Amount', 'Fraud Probability']].sort_values('Fraud Probability', ascending=False).head(10))
        except Exception as e:
            st.error(f"Processing error: {str(e)}")
    st.markdown('</div>', unsafe_allow_html=True)
