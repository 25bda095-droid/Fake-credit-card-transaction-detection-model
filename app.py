

# import streamlit as st
# import pandas as pd
# import numpy as np
# import joblib
# import PyPDF2
# import re
# import io
# import os
# from PIL import Image

# # ---- LOAD MODELS AND SCALER ----
# MODELPATH = 'fraud_detection_model_tuned.pkl'
# SCALERPATH = 'scaler.pkl'
# RANDOMFORESTPATH = 'random_Forest_model.pkl'
# XGBOOSTPATH = 'XGBoost_model.joblib'

# @st.cache_resource
# def load_models():
#     try:
#         tunedmodel = joblib.load(MODELPATH)
#         scaler = joblib.load(SCALERPATH)
#         rfmodel = joblib.load(RANDOMFORESTPATH)
#         xgbmodel = joblib.load(XGBOOSTPATH)
#         return tunedmodel, scaler, rfmodel, xgbmodel
#     except Exception as e:
#         st.error(f"Error loading models: {e}")
#         return None, None, None, None

# tunedmodel, scaler, rfmodel, xgbmodel = load_models()

# ALLOWED_EXTENSIONS = ['csv', 'pdf']

# def allowed_file(filename):
#     return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# def extract_csv_from_pdf(pdf_file):
#     try:
#         pdfreader = PyPDF2.PdfReader(pdf_file)
#         text = ''
#         for page in pdfreader.pages:
#             text += page.extract_text()
#         lines = text.strip().split('\n')
#         data = []
#         for line in lines:
#             row = re.split(r',\s*', line.strip())
#             if len(row) > 1:
#                 data.append(row)
#         if data:
#             df = pd.DataFrame(data[1:], columns=data[0])
#             return df
#         else:
#             return None
#     except Exception as e:
#         st.error(f"PDF extraction error: {e}")
#         return None

# def preprocess_data(df):
#     try:
#         required_cols = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']
#         missing_cols = [col for col in required_cols if col not in df.columns]
#         if missing_cols:
#             if 'Time' in missing_cols:
#                 required_cols = [f'V{i}' for i in range(1, 29)] + ['Amount']
#                 missing_cols = [col for col in required_cols if col not in df.columns]
#             if missing_cols:
#                 return None, f"Missing required columns: {', '.join(missing_cols)}"
#         for col in required_cols:
#             df[col] = pd.to_numeric(df[col], errors='coerce')
#         df = df.dropna(subset=required_cols)
#         if len(df) == 0:
#             return None, "No valid data found after cleaning."
#         feature_cols = required_cols
#         X = df[feature_cols].copy()
#         X_scaled = scaler.transform(X)
#         return X_scaled, None
#     except Exception as e:
#         return None, f"Preprocessing error: {str(e)}"

# def predict_fraud(X_scaled, model_choice):
#     try:
#         if model_choice == "tuned":
#             model = tunedmodel
#         elif model_choice == "rf":
#             model = rfmodel
#         elif model_choice == "xgb":
#             model = xgbmodel
#         else:
#             st.error("Unknown model selected.")
#             return None, None
#         predictions = model.predict(X_scaled)
#         probabilities = model.predict_proba(X_scaled)[:, 1]
#         return predictions, probabilities
#     except Exception as e:
#         st.error(f"Prediction error: {e}")
#         return None, None

# # # ---- STREAMLIT APP UI ----

# # st.markdown("""
# # <div style='display:flex;flex-direction:column;align-items:center;justify-content:center;margin-top:2em;'>
# #   <img src='https://cdn-icons-png.flaticon.com/512/3135/3135715.png' width='90' style='margin-bottom:1em;border-radius:50%;box-shadow:0 2px 12px #ffb80055;'>
# #   <h1 style='margin-bottom:0.2em;font-size:2.8rem;font-weight:800;letter-spacing:1px;'>Fraud Detection System</h1>
# #   <h4 style='color:#ffb800;font-size:1.3rem;font-weight:600;margin-bottom:1.2em;'>CREDIT CARD TRANSACTION ANALYSIS</h4>
# # </div>
# # """, unsafe_allow_html=True)

# # st.markdown("""
# # <div style='background:#232946;border-radius:18px;padding:22px 32px;margin-bottom:22px;color:#f4f4f4;box-shadow:0 4px 24px rgba(0,0,0,0.13);max-width:600px;margin-left:auto;margin-right:auto;'>
# # <b style='font-size:1.1em;color:#ffb800;'>How to Use:</b><br><br>
# # <ul style='font-size:1.08em;line-height:1.7;'>
# # <li>Upload your credit card transaction data (<b>CSV</b> or <b>PDF</b>).</li>
# # <li>File must have columns: <b>Time, V1-V28, Amount</b>.</li>
# # <li>Select your preferred ML model for analysis.</li>
# # <li>Click <b>Analyze Transactions</b> to get instant fraud detection results!</li>
# # </ul>
# # <span style='color:#ff4b4b;'>Your data is processed securely and never stored.</span>
# # </div>
# # """, unsafe_allow_html=True)

# # uploaded_file = st.file_uploader("Choose your file (CSV or PDF)", type=ALLOWED_EXTENSIONS)
# # model_choice = st.radio(
# #     "Select Model",
# #     [
# #         ("Tuned Model", "tuned"),
# #         ("Random Forest Model", "rf"),
# #         ("XGBoost Model", "xgb")
# #     ],
# #     format_func=lambda x: x[0]
# # )[1]
# # analyze = st.button("Analyze Transactions")

# # if analyze and uploaded_file is not None:
# #     filename = uploaded_file.name
# #     if not allowed_file(filename):
# #         st.error("Invalid file format. Please upload a CSV or PDF.")
# #     else:
# #         try:
# #             if filename.endswith('.csv'):
# #                 df = pd.read_csv(uploaded_file)
# #             else:
# #                 df = extract_csv_from_pdf(uploaded_file)
# #             if df is None:
# #                 st.error("Could not extract data from the uploaded file.")
# #             else:
# #                 X_scaled, error = preprocess_data(df)
# #                 if error:
# #                     st.error(error)
# #                 else:
# #                     predictions, probabilities = predict_fraud(X_scaled, model_choice)
# #                     if predictions is None:
# #                         st.error("Prediction failed.")
# #                     else:
# #                         total_transactions = len(predictions)
# #                         fraud_count = int(np.sum(predictions))
# #                         legitimate_count = total_transactions - fraud_count
# #                         fraud_percentage = (fraud_count / total_transactions) * 100
# #                         st.markdown(f"<div style='text-align:center;margin-bottom:18px;'><b>Total Transactions:</b> {total_transactions}</div>", unsafe_allow_html=True)
# #                         st.markdown(f"<div style='color:#fc5c7d;font-weight:600;'>Fraudulent: {fraud_count} ({fraud_percentage:.2f}%)</div>", unsafe_allow_html=True)
# #                         st.markdown(f"<div style='color:#6a82fb;font-weight:600;'>Legitimate: {legitimate_count}</div>", unsafe_allow_html=True)
# #                         # High Risk Transactions
# #                         st.markdown("<hr style='margin:18px 0;'>", unsafe_allow_html=True)
# #                         st.markdown("<div style='font-weight:600;font-size:1.1rem;margin-bottom:8px;'>High-Risk Transactions (Prob > 0.70)</div>", unsafe_allow_html=True)

# #                         # --- Charts Section ---
# #                         import matplotlib.pyplot as plt
# #                         import seaborn as sns
# #                         import pandas as pd
# #                         # Pie chart for fraud vs legit
# #                         fig1, ax1 = plt.subplots(figsize=(3,3))
# #                         ax1.pie([fraud_count, legitimate_count], labels=['Fraudulent', 'Legitimate'], autopct='%1.1f%%', colors=['#fc5c7d', '#6a82fb'], startangle=90, textprops={'color':'#232946','fontweight':'bold'})
# #                         ax1.set_title('Fraud vs Legitimate', fontsize=14, color='#232946')
# #                         st.pyplot(fig1)

# #                         # Bar chart for probability distribution
# #                         fig2, ax2 = plt.subplots(figsize=(5,2.5))
# #                         sns.histplot(probabilities, bins=20, kde=True, color='#6a82fb', ax=ax2)
# #                         ax2.set_xlabel('Fraud Probability')
# #                         ax2.set_ylabel('Count')
# #                         ax2.set_title('Probability Distribution of Transactions', color='#232946')
# #                         st.pyplot(fig2)

# #                         # If you want a table of high-risk transactions:
# #                         high_risk_indices = np.where(probabilities > 0.70)[0]
# #                         if len(high_risk_indices) > 0:
# #                             st.markdown('<b>High-Risk Transactions Table</b>', unsafe_allow_html=True)
# #                             high_risk_df = df.iloc[high_risk_indices].copy()
# #                             high_risk_df['Fraud Probability'] = probabilities[high_risk_indices]
# #                             st.dataframe(high_risk_df[['Amount', 'Fraud Probability']].sort_values('Fraud Probability', ascending=False).head(10))
# #         except Exception as e:
# #             st.error(f"Processing error: {str(e)}")
# #     st.markdown('</div>', unsafe_allow_html=True)


# # ---- STREAMLIT APP UI ----

# st.markdown("""
# <div style='display:flex;flex-direction:column;align-items:center;justify-content:center;margin-top:2em;'>
#   <img src='https://cdn-icons-png.flaticon.com/512/3135/3135715.png' width='90' style='margin-bottom:1em;border-radius:50%;box-shadow:0 2px 12px #ffb80055;'>
#   <h1 style='margin-bottom:0.2em;font-size:2.8rem;font-weight:800;letter-spacing:1px;'>Fraud Detection System</h1>
#   <h4 style='color:#ffb800;font-size:1.3rem;font-weight:600;margin-bottom:1.2em;'>CREDIT CARD TRANSACTION ANALYSIS</h4>
# </div>
# """, unsafe_allow_html=True)

# st.markdown("""
# <div style='background:#232946;border-radius:18px;padding:22px 32px;margin-bottom:22px;color:#f4f4f4;box-shadow:0 4px 24px rgba(0,0,0,0.13);max-width:600px;margin-left:auto;margin-right:auto;'>
# <b style='font-size:1.1em;color:#ffb800;'>How to Use:</b><br><br>
# <ul style='font-size:1.08em;line-height:1.7;'>
# <li>Upload your credit card transaction data (<b>CSV</b> or <b>PDF</b>).</li>
# <li>File must have columns: <b>Time, V1-V28, Amount</b>.</li>
# <li>Click <b>Check Transaction</b> to analyze with all 3 models.</li>
# <li>Get instant fraud detection results with model comparison!</li>
# </ul>
# <span style='color:#ff4b4b;'>Your data is processed securely and never stored.</span>
# </div>
# """, unsafe_allow_html=True)

# uploaded_file = st.file_uploader("Choose your file (CSV or PDF)", type=ALLOWED_EXTENSIONS)
# check = st.button("Check Transaction")

# if check and uploaded_file is not None:
#     filename = uploaded_file.name
#     if not allowed_file(filename):
#         st.error("Invalid file format. Please upload a CSV or PDF.")
#     else:
#         try:
#             if filename.endswith('.csv'):
#                 df = pd.read_csv(uploaded_file)
#             else:
#                 df = extract_csv_from_pdf(uploaded_file)
#             if df is None:
#                 st.error("Could not extract data from the uploaded file.")
#             else:
#                 X_scaled, error = preprocess_data(df)
#                 if error:
#                     st.error(error)
#                 else:
#                     # Get predictions from all three models
#                     pred_tuned, prob_tuned = predict_fraud(X_scaled, "tuned")
#                     pred_rf, prob_rf = predict_fraud(X_scaled, "rf")
#                     pred_xgb, prob_xgb = predict_fraud(X_scaled, "xgb")
                    
#                     if pred_tuned is None or pred_rf is None or pred_xgb is None:
#                         st.error("Prediction failed.")
#                     else:
#                         total_transactions = len(pred_tuned)
                        
#                         # Overall Summary
#                         st.markdown("<div style='font-weight:600;font-size:1.3rem;margin-bottom:12px;'>Overall Summary</div>", unsafe_allow_html=True)
#                         col1, col2, col3 = st.columns(3)
                        
#                         with col1:
#                             st.metric("Total Transactions", total_transactions)
                        
#                         with col2:
#                             tuned_fraud = int(np.sum(pred_tuned))
#                             st.metric("Tuned Model - Fraud", tuned_fraud)
                        
#                         with col3:
#                             st.metric("Average Fraud Probability", f"{(np.mean(prob_tuned) + np.mean(prob_rf) + np.mean(prob_xgb)) / 3:.2%}")
                        
#                         st.markdown("<hr style='margin:18px 0;'>", unsafe_allow_html=True)
                        
#                         # Model Comparison Table
#                         st.markdown("<div style='font-weight:600;font-size:1.1rem;margin-bottom:12px;'>Model Comparison Results</div>", unsafe_allow_html=True)
                        
#                         comparison_data = {
#                             'Model': ['Tuned Model', 'Random Forest', 'XGBoost'],
#                             'Fraudulent Count': [
#                                 int(np.sum(pred_tuned)),
#                                 int(np.sum(pred_rf)),
#                                 int(np.sum(pred_xgb))
#                             ],
#                             'Legitimate Count': [
#                                 total_transactions - int(np.sum(pred_tuned)),
#                                 total_transactions - int(np.sum(pred_rf)),
#                                 total_transactions - int(np.sum(pred_xgb))
#                             ],
#                             'Fraud %': [
#                                 f"{(int(np.sum(pred_tuned)) / total_transactions) * 100:.2f}%",
#                                 f"{(int(np.sum(pred_rf)) / total_transactions) * 100:.2f}%",
#                                 f"{(int(np.sum(pred_xgb)) / total_transactions) * 100:.2f}%"
#                             ],
#                             'Avg Fraud Probability': [
#                                 f"{np.mean(prob_tuned):.4f}",
#                                 f"{np.mean(prob_rf):.4f}",
#                                 f"{np.mean(prob_xgb):.4f}"
#                             ]
#                         }
                        
#                         comparison_df = pd.DataFrame(comparison_data)
#                         st.dataframe(comparison_df, use_container_width=True, hide_index=True)
                        
#                         st.markdown("<hr style='margin:18px 0;'>", unsafe_allow_html=True)
                        
#                         # Charts Section
#                         st.markdown("<div style='font-weight:600;font-size:1.1rem;margin-bottom:12px;'>Visualization</div>", unsafe_allow_html=True)
                        
#                         import matplotlib.pyplot as plt
#                         import seaborn as sns
                        
#                         col1, col2 = st.columns(2)
                        
#                         with col1:
#                             # Pie charts for each model
#                             fig1, axes = plt.subplots(1, 3, figsize=(15, 4))
                            
#                             models_data = [
#                                 ('Tuned Model', int(np.sum(pred_tuned)), total_transactions - int(np.sum(pred_tuned))),
#                                 ('Random Forest', int(np.sum(pred_rf)), total_transactions - int(np.sum(pred_rf))),
#                                 ('XGBoost', int(np.sum(pred_xgb)), total_transactions - int(np.sum(pred_xgb)))
#                             ]
                            
#                             for idx, (model_name, fraud_count, legit_count) in enumerate(models_data):
#                                 axes[idx].pie([fraud_count, legit_count], labels=['Fraudulent', 'Legitimate'], 
#                                              autopct='%1.1f%%', colors=['#fc5c7d', '#6a82fb'], startangle=90, 
#                                              textprops={'color':'#232946','fontweight':'bold'})
#                                 axes[idx].set_title(model_name, fontsize=12, color='#232946', fontweight='bold')
                            
#                             plt.tight_layout()
#                             st.pyplot(fig1)
                        
#                         with col2:
#                             # Probability distribution comparison
#                             fig2, ax2 = plt.subplots(figsize=(10, 5))
                            
#                             ax2.hist(prob_tuned, bins=20, alpha=0.5, label='Tuned Model', color='#fc5c7d')
#                             ax2.hist(prob_rf, bins=20, alpha=0.5, label='Random Forest', color='#6a82fb')
#                             ax2.hist(prob_xgb, bins=20, alpha=0.5, label='XGBoost', color='#ffb800')
                            
#                             ax2.set_xlabel('Fraud Probability', fontsize=11, fontweight='bold')
#                             ax2.set_ylabel('Count', fontsize=11, fontweight='bold')
#                             ax2.set_title('Probability Distribution - Model Comparison', fontsize=12, color='#232946', fontweight='bold')
#                             ax2.legend()
#                             ax2.grid(True, alpha=0.3)
                            
#                             st.pyplot(fig2)
                        
#                         st.markdown("<hr style='margin:18px 0;'>", unsafe_allow_html=True)
                        
#                         # High-Risk Transactions Consensus
#                         st.markdown("<div style='font-weight:600;font-size:1.1rem;margin-bottom:12px;'>High-Risk Transactions (Consensus)</div>", unsafe_allow_html=True)
                        
#                         # Transactions flagged by at least 2 out of 3 models
#                         consensus_fraud = (pred_tuned.astype(int) + pred_rf.astype(int) + pred_xgb.astype(int)) >= 2
#                         avg_prob = (prob_tuned + prob_rf + prob_xgb) / 3
                        
#                         high_risk_indices = np.where(consensus_fraud)[0]
                        
#                         if len(high_risk_indices) > 0:
#                             high_risk_df = df.iloc[high_risk_indices].copy()
#                             high_risk_df['Tuned Prob'] = prob_tuned[high_risk_indices]
#                             high_risk_df['RF Prob'] = prob_rf[high_risk_indices]
#                             high_risk_df['XGB Prob'] = prob_xgb[high_risk_indices]
#                             high_risk_df['Avg Probability'] = avg_prob[high_risk_indices]
#                             high_risk_df['Models Flagged'] = (pred_tuned[high_risk_indices].astype(int) + 
#                                                               pred_rf[high_risk_indices].astype(int) + 
#                                                               pred_xgb[high_risk_indices].astype(int))
                            
#                             display_cols = ['Amount', 'Tuned Prob', 'RF Prob', 'XGB Prob', 'Avg Probability', 'Models Flagged']
#                             st.dataframe(high_risk_df[display_cols].sort_values('Avg Probability', ascending=False).head(15), 
#                                         use_container_width=True, hide_index=True)
#                         else:
#                             st.info("No high-risk transactions detected by consensus.")
                        
#         except Exception as e:
#             st.error(f"Processing error: {str(e)}")
#     st.markdown('</div>', unsafe_allow_html=True)

# ---- STREAMLIT APP UI ----

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
<li>Click <b>Check Transaction</b> to analyze with all 3 models.</li>
<li>Get instant fraud detection results with model comparison!</li>
</ul>
<span style='color:#ff4b4b;'>Your data is processed securely and never stored.</span>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Choose your file (CSV or PDF)", type=ALLOWED_EXTENSIONS)
check = st.button("🔍 Check Transaction", use_container_width=True)

if check and uploaded_file is not None:
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
                    # Get predictions from all three models
                    pred_tuned, prob_tuned = predict_fraud(X_scaled, "tuned")
                    pred_rf, prob_rf = predict_fraud(X_scaled, "rf")
                    pred_xgb, prob_xgb = predict_fraud(X_scaled, "xgb")
                    
                    if pred_tuned is None or pred_rf is None or pred_xgb is None:
                        st.error("Prediction failed.")
                    else:
                        total_transactions = len(pred_tuned)
                        
                        # Overall Summary
                        st.markdown("<div style='font-weight:700;font-size:1.5rem;margin-top:20px;margin-bottom:16px;color:#232946;'>📊 Overall Summary</div>", unsafe_allow_html=True)
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("📈 Total Transactions", total_transactions, delta=None)
                        
                        with col2:
                            tuned_fraud = int(np.sum(pred_tuned))
                            st.metric("🚨 Tuned Model Fraud", tuned_fraud)
                        
                        with col3:
                            rf_fraud = int(np.sum(pred_rf))
                            st.metric("🌳 RF Model Fraud", rf_fraud)
                        
                        with col4:
                            xgb_fraud = int(np.sum(pred_xgb))
                            st.metric("⚡ XGB Model Fraud", xgb_fraud)
                        
                        st.markdown("<hr style='margin:20px 0;border:2px solid #ffb800;'>", unsafe_allow_html=True)
                        
                        # Model Comparison Table
                        st.markdown("<div style='font-weight:700;font-size:1.4rem;margin-bottom:16px;color:#232946;'>🔄 Model Comparison</div>", unsafe_allow_html=True)
                        
                        comparison_data = {
                            'Model': ['🚨 Tuned Model', '🌳 Random Forest', '⚡ XGBoost'],
                            'Fraudulent': [
                                int(np.sum(pred_tuned)),
                                int(np.sum(pred_rf)),
                                int(np.sum(pred_xgb))
                            ],
                            'Legitimate': [
                                total_transactions - int(np.sum(pred_tuned)),
                                total_transactions - int(np.sum(pred_rf)),
                                total_transactions - int(np.sum(pred_xgb))
                            ],
                            'Fraud %': [
                                f"{(int(np.sum(pred_tuned)) / total_transactions) * 100:.2f}%",
                                f"{(int(np.sum(pred_rf)) / total_transactions) * 100:.2f}%",
                                f"{(int(np.sum(pred_xgb)) / total_transactions) * 100:.2f}%"
                            ],
                            'Avg Probability': [
                                f"{np.mean(prob_tuned):.4f}",
                                f"{np.mean(prob_rf):.4f}",
                                f"{np.mean(prob_xgb):.4f}"
                            ]
                        }
                        
                        comparison_df = pd.DataFrame(comparison_data)
                        st.dataframe(comparison_df, use_container_width=True, hide_index=True)
                        
                        st.markdown("<hr style='margin:20px 0;border:2px solid #ffb800;'>", unsafe_allow_html=True)
                        
                        # Charts Section
                        st.markdown("<div style='font-weight:700;font-size:1.4rem;margin-bottom:16px;color:#232946;'>📈 Visualizations</div>", unsafe_allow_html=True)
                        
                        import matplotlib.pyplot as plt
                        import seaborn as sns
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Pie charts for each model
                            fig1, axes = plt.subplots(1, 3, figsize=(16, 5))
                            fig1.patch.set_facecolor('#f8f9fa')
                            
                            models_data = [
                                ('🚨 Tuned Model', int(np.sum(pred_tuned)), total_transactions - int(np.sum(pred_tuned))),
                                ('🌳 Random Forest', int(np.sum(pred_rf)), total_transactions - int(np.sum(pred_rf))),
                                ('⚡ XGBoost', int(np.sum(pred_xgb)), total_transactions - int(np.sum(pred_xgb)))
                            ]
                            
                            for idx, (model_name, fraud_count, legit_count) in enumerate(models_data):
                                axes[idx].pie([fraud_count, legit_count], labels=['Fraudulent', 'Legitimate'], 
                                             autopct='%1.1f%%', colors=['#fc5c7d', '#6a82fb'], startangle=90, 
                                             textprops={'color':'#232946','fontweight':'bold', 'fontsize': 11})
                                axes[idx].set_title(model_name, fontsize=13, color='#232946', fontweight='bold', pad=15)
                            
                            plt.tight_layout()
                            st.pyplot(fig1)
                        
                        with col2:
                            # Probability distribution comparison
                            fig2, ax2 = plt.subplots(figsize=(11, 5))
                            fig2.patch.set_facecolor('#f8f9fa')
                            
                            ax2.hist(prob_tuned, bins=25, alpha=0.6, label='🚨 Tuned Model', color='#fc5c7d', edgecolor='black', linewidth=1.2)
                            ax2.hist(prob_rf, bins=25, alpha=0.6, label='🌳 Random Forest', color='#6a82fb', edgecolor='black', linewidth=1.2)
                            ax2.hist(prob_xgb, bins=25, alpha=0.6, label='⚡ XGBoost', color='#ffb800', edgecolor='black', linewidth=1.2)
                            
                            ax2.set_xlabel('Fraud Probability', fontsize=12, fontweight='bold')
                            ax2.set_ylabel('Count', fontsize=12, fontweight='bold')
                            ax2.set_title('Probability Distribution - Model Comparison', fontsize=13, color='#232946', fontweight='bold', pad=15)
                            ax2.legend(fontsize=11, loc='upper right')
                            ax2.grid(True, alpha=0.3, linestyle='--')
                            ax2.set_axisbelow(True)
                            
                            st.pyplot(fig2)
                        
                        st.markdown("<hr style='margin:20px 0;border:2px solid #ffb800;'>", unsafe_allow_html=True)
                        
                        # Calculate consensus levels
                        models_count = (pred_tuned.astype(int) + pred_rf.astype(int) + pred_xgb.astype(int))
                        avg_prob = (prob_tuned + prob_rf + prob_xgb) / 3
                        
                        fraud_3_models = np.where(models_count == 3)[0]
                        fraud_2_models = np.where(models_count == 2)[0]
                        fraud_1_model = np.where(models_count == 1)[0]
                        
                        # 100% Fraud (All 3 Models Detected)
                        st.markdown("<div style='background:#fc5c7d;border-radius:15px;padding:15px;margin-bottom:20px;box-shadow:0 4px 12px rgba(252, 92, 125, 0.3);'><div style='font-weight:700;font-size:1.3rem;color:white;'>🔴 100% FRAUD - Detected by All 3 Models</div></div>", unsafe_allow_html=True)
                        
                        if len(fraud_3_models) > 0:
                            fraud_3_df = df.iloc[fraud_3_models].copy()
                            fraud_3_df['Tuned Prob'] = prob_tuned[fraud_3_models]
                            fraud_3_df['RF Prob'] = prob_rf[fraud_3_models]
                            fraud_3_df['XGB Prob'] = prob_xgb[fraud_3_models]
                            fraud_3_df['Avg Probability'] = avg_prob[fraud_3_models]
                            
                            display_cols = ['Amount', 'Tuned Prob', 'RF Prob', 'XGB Prob', 'Avg Probability']
                            st.dataframe(fraud_3_df[display_cols].sort_values('Avg Probability', ascending=False).round(4), 
                                        use_container_width=True, hide_index=True)
                            st.success(f"✅ Found {len(fraud_3_models)} HIGH CONFIDENCE fraud transactions!")
                        else:
                            st.info("ℹ️ No transactions detected as fraud by all 3 models.")
                        
                        st.markdown("<hr style='margin:20px 0;'>", unsafe_allow_html=True)
                        
                        # 2 Models Detected
                        st.markdown("<div style='background:#ffb800;border-radius:15px;padding:15px;margin-bottom:20px;box-shadow:0 4px 12px rgba(255, 184, 0, 0.3);'><div style='font-weight:700;font-size:1.3rem;color:white;'>🟠 MEDIUM RISK - Detected by 2 Models</div></div>", unsafe_allow_html=True)
                        
                        if len(fraud_2_models) > 0:
                            fraud_2_df = df.iloc[fraud_2_models].copy()
                            fraud_2_df['Tuned Prob'] = prob_tuned[fraud_2_models]
                            fraud_2_df['RF Prob'] = prob_rf[fraud_2_models]
                            fraud_2_df['XGB Prob'] = prob_xgb[fraud_2_models]
                            fraud_2_df['Avg Probability'] = avg_prob[fraud_2_models]
                            fraud_2_df['Models Flagged'] = models_count[fraud_2_models]
                            
                            display_cols = ['Amount', 'Tuned Prob', 'RF Prob', 'XGB Prob', 'Avg Probability', 'Models Flagged']
                            st.dataframe(fraud_2_df[display_cols].sort_values('Avg Probability', ascending=False).round(4), 
                                        use_container_width=True, hide_index=True)
                            st.warning(f"⚠️ Found {len(fraud_2_models)} MEDIUM RISK fraud transactions!")
                        else:
                            st.info("ℹ️ No transactions detected as fraud by exactly 2 models.")
                        
                        st.markdown("<hr style='margin:20px 0;'>", unsafe_allow_html=True)
                        
                        # 1 Model Detected
                        st.markdown("<div style='background:#6a82fb;border-radius:15px;padding:15px;margin-bottom:20px;box-shadow:0 4px 12px rgba(106, 130, 251, 0.3);'><div style='font-weight:700;font-size:1.3rem;color:white;'>🟡 LOW RISK - Detected by 1 Model</div></div>", unsafe_allow_html=True)
                        
                        if len(fraud_1_model) > 0:
                            fraud_1_df = df.iloc[fraud_1_model].copy()
                            fraud_1_df['Tuned Prob'] = prob_tuned[fraud_1_model]
                            fraud_1_df['RF Prob'] = prob_rf[fraud_1_model]
                            fraud_1_df['XGB Prob'] = prob_xgb[fraud_1_model]
                            fraud_1_df['Avg Probability'] = avg_prob[fraud_1_model]
                            fraud_1_df['Models Flagged'] = models_count[fraud_1_model]
                            
                            display_cols = ['Amount', 'Tuned Prob', 'RF Prob', 'XGB Prob', 'Avg Probability', 'Models Flagged']
                            st.dataframe(fraud_1_df[display_cols].sort_values('Avg Probability', ascending=False).round(4), 
                                        use_container_width=True, hide_index=True)
                            st.info(f"ℹ️ Found {len(fraud_1_model)} LOW RISK fraud transactions!")
                        else:
                            st.info("ℹ️ No transactions detected as fraud by exactly 1 model.")
                        
                        st.markdown("<hr style='margin:20px 0;border:2px solid #ffb800;'>", unsafe_allow_html=True)
                        
                        # Summary Statistics
                        st.markdown("<div style='font-weight:700;font-size:1.4rem;margin-bottom:16px;color:#232946;'>📋 Risk Summary</div>", unsafe_allow_html=True)
                        
                        summary_col1, summary_col2, summary_col3 = st.columns(3)
                        
                        with summary_col1:
                            st.metric("🔴 100% Fraud (3/3)", len(fraud_3_models), delta=None)
                        
                        with summary_col2:
                            st.metric("🟠 Medium Risk (2/3)", len(fraud_2_models), delta=None)
                        
                        with summary_col3:
                            st.metric("🟡 Low Risk (1/3)", len(fraud_1_model), delta=None)
                        
        except Exception as e:
            st.error(f"Processing error: {str(e)}")
