

# import streamlit as st
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# import joblib
# import PyPDF2
# import re
# from sklearn.metrics import roc_curve, auc, recall_score

# # Define allowed extensions
# ALLOWED_EXTENSIONS = ['csv', 'pdf']
# DECISION_TREE_PATH = 'decision_tree_model.joblib'
# RANDOM_FOREST_PATH = 'random_Forest_model.pkl'
# XGBOOST_PATH = 'XGBoost_model.joblib'
# LOGISTIC_REGRESSION_PATH = 'logistic_regression_model.joblib'
# SCALER_PATH = 'scaler.pkl'
# SCALER_JOBLIB_PATH = 'scaler.joblib'

# @st.cache_resource
# def load_models():
#     try:
#         dt_model = joblib.load(DECISION_TREE_PATH)
#         rf_model = joblib.load(RANDOM_FOREST_PATH)
#         xgb_model = joblib.load(XGBOOST_PATH)
#         lr_model = joblib.load(LOGISTIC_REGRESSION_PATH)
        
#         # Try to load scaler.pkl first, if not found try scaler.joblib
#         try:
#             scaler = joblib.load(SCALER_PATH)
#         except:
#             scaler = joblib.load(SCALER_JOBLIB_PATH)
        
#         return dt_model, rf_model, xgb_model, lr_model, scaler
#     except Exception as e:
#         st.error(f"Error loading models: {e}")
#         return None, None, None, None, None

# dt_model, rf_model, xgb_model, lr_model, scaler = load_models()

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
#                 return None, None, f"Missing required columns: {', '.join(missing_cols)}"
#         for col in required_cols:
#             df[col] = pd.to_numeric(df[col], errors='coerce')
#         df = df.dropna(subset=required_cols)
#         if len(df) == 0:
#             return None, None, "No valid data found after cleaning."
        
#         # Check if Class column exists for true labels
#         y_true = None
#         if 'Class' in df.columns:
#             df['Class'] = pd.to_numeric(df['Class'], errors='coerce')
#             y_true = df['Class'].values
        
#         feature_cols = required_cols
#         X = df[feature_cols].copy()
#         X_scaled = scaler.transform(X)
#         return X_scaled, y_true, None
#     except Exception as e:
#         return None, None, f"Preprocessing error: {str(e)}"

# def predict_fraud(X_scaled, model_choice):
#     try:
#         if model_choice == "dt":
#             model = dt_model
#         elif model_choice == "rf":
#             model = rf_model
#         elif model_choice == "xgb":
#             model = xgb_model
#         elif model_choice == "lr":
#             model = lr_model
#         else:
#             st.error("Unknown model selected.")
#             return None, None
#         predictions = model.predict(X_scaled)
#         probabilities = model.predict_proba(X_scaled)[:, 1]
#         return predictions, probabilities
#     except Exception as e:
#         st.error(f"Prediction error: {e}")
#         return None, None

# st.set_page_config(page_title="Fraud Detection", layout="wide")

# st.title("Fraud Detection System")
# st.markdown("### CREDIT CARD TRANSACTION ANALYSIS")

# st.info("How to Use:\n- Upload your credit card transaction data (CSV or PDF)\n- File must have columns: Time, V1-V28, Amount\n- Optional: Include 'Class' column (0=Legitimate, 1=Fraud) for ROC AUC and Recall metrics\n- Click Check Transaction to analyze with all 4 models\n- Get instant fraud detection results with model comparison!")

# uploaded_file = st.file_uploader("Choose your file (CSV or PDF)", type=ALLOWED_EXTENSIONS)
# check = st.button("Check Transaction", use_container_width=True)

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
#                 X_scaled, y_true, error = preprocess_data(df)
#                 if error:
#                     st.error(error)
#                 else:
#                     pred_dt, prob_dt = predict_fraud(X_scaled, "dt")
#                     pred_rf, prob_rf = predict_fraud(X_scaled, "rf")
#                     pred_xgb, prob_xgb = predict_fraud(X_scaled, "xgb")
#                     pred_lr, prob_lr = predict_fraud(X_scaled, "lr")
                    
#                     if pred_dt is None or pred_rf is None or pred_xgb is None or pred_lr is None:
#                         st.error("Prediction failed.")
#                     else:
#                         total_transactions = len(pred_dt)
                        
#                         st.subheader("Overall Summary")
#                         col1, col2, col3, col4, col5 = st.columns(5)
                        
#                         with col1:
#                             st.metric("Total Transactions", total_transactions)
                        
#                         with col2:
#                             st.metric("Decision Tree Fraud", int(np.sum(pred_dt)))
                        
#                         with col3:
#                             st.metric("Random Forest Fraud", int(np.sum(pred_rf)))
                        
#                         with col4:
#                             st.metric("XGBoost Fraud", int(np.sum(pred_xgb)))
                        
#                         with col5:
#                             st.metric("Logistic Regression Fraud", int(np.sum(pred_lr)))
                        
#                         st.divider()
                        
#                         # Calculate metrics if true labels are available
#                         if y_true is not None:
#                             st.subheader("Model Performance Metrics")
                            
#                             recall_dt = recall_score(y_true, pred_dt)
#                             recall_rf = recall_score(y_true, pred_rf)
#                             recall_xgb = recall_score(y_true, pred_xgb)
#                             recall_lr = recall_score(y_true, pred_lr)
                            
#                             fpr_dt, tpr_dt, _ = roc_curve(y_true, prob_dt)
#                             fpr_rf, tpr_rf, _ = roc_curve(y_true, prob_rf)
#                             fpr_xgb, tpr_xgb, _ = roc_curve(y_true, prob_xgb)
#                             fpr_lr, tpr_lr, _ = roc_curve(y_true, prob_lr)
                            
#                             roc_auc_dt = auc(fpr_dt, tpr_dt)
#                             roc_auc_rf = auc(fpr_rf, tpr_rf)
#                             roc_auc_xgb = auc(fpr_xgb, tpr_xgb)
#                             roc_auc_lr = auc(fpr_lr, tpr_lr)
                            
#                             col1, col2 = st.columns(2)
                            
#                             with col1:
#                                 fig_roc, ax_roc = plt.subplots(figsize=(10, 8))
#                                 fig_roc.patch.set_facecolor('white')
                                
#                                 ax_roc.plot(fpr_dt, tpr_dt, color='#2ecc71', lw=2, 
#                                            label=f'Decision Tree (AUC = {roc_auc_dt:.4f})')
#                                 ax_roc.plot(fpr_rf, tpr_rf, color='#6a82fb', lw=2, 
#                                            label=f'Random Forest (AUC = {roc_auc_rf:.4f})')
#                                 ax_roc.plot(fpr_xgb, tpr_xgb, color='#ffb800', lw=2, 
#                                            label=f'XGBoost (AUC = {roc_auc_xgb:.4f})')
#                                 ax_roc.plot(fpr_lr, tpr_lr, color='#fc5c7d', lw=2, 
#                                            label=f'Logistic Regression (AUC = {roc_auc_lr:.4f})')
#                                 ax_roc.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', label='Random Classifier')
                                
#                                 ax_roc.set_xlim([0.0, 1.0])
#                                 ax_roc.set_ylim([0.0, 1.05])
#                                 ax_roc.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
#                                 ax_roc.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
#                                 ax_roc.set_title('ROC Curve - Model Comparison', fontsize=14, fontweight='bold')
#                                 ax_roc.legend(loc="lower right", fontsize=9)
#                                 ax_roc.grid(True, alpha=0.3)
                                
#                                 st.pyplot(fig_roc)
                            
#                             with col2:
#                                 fig_recall, ax_recall = plt.subplots(figsize=(10, 8))
#                                 fig_recall.patch.set_facecolor('white')
                                
#                                 models = ['Decision Tree', 'Random Forest', 'XGBoost', 'Logistic Reg']
#                                 recalls = [recall_dt, recall_rf, recall_xgb, recall_lr]
#                                 colors = ['#2ecc71', '#6a82fb', '#ffb800', '#fc5c7d']
                                
#                                 bars = ax_recall.bar(models, recalls, color=colors, edgecolor='black', linewidth=1.5)
                                
#                                 for bar, recall in zip(bars, recalls):
#                                     height = bar.get_height()
#                                     ax_recall.text(bar.get_x() + bar.get_width()/2., height,
#                                                   f'{recall:.4f}',
#                                                   ha='center', va='bottom', fontsize=11, fontweight='bold')
                                
#                                 ax_recall.set_ylim([0, 1.1])
#                                 ax_recall.set_ylabel('Recall Score', fontsize=12, fontweight='bold')
#                                 ax_recall.set_title('Recall Score - Model Comparison', fontsize=14, fontweight='bold')
#                                 ax_recall.grid(True, axis='y', alpha=0.3)
#                                 plt.xticks(rotation=15, ha='right')
                                
#                                 st.pyplot(fig_recall)
                            
#                             st.divider()
                        
#                         st.subheader("Model Comparison")
#                         comparison_data = {
#                             'Model': ['Decision Tree', 'Random Forest', 'XGBoost', 'Logistic Regression'],
#                             'Fraudulent': [
#                                 int(np.sum(pred_dt)),
#                                 int(np.sum(pred_rf)),
#                                 int(np.sum(pred_xgb)),
#                                 int(np.sum(pred_lr))
#                             ],
#                             'Legitimate': [
#                                 total_transactions - int(np.sum(pred_dt)),
#                                 total_transactions - int(np.sum(pred_rf)),
#                                 total_transactions - int(np.sum(pred_xgb)),
#                                 total_transactions - int(np.sum(pred_lr))
#                             ],
#                             'Fraud %': [
#                                 f"{(int(np.sum(pred_dt)) / total_transactions) * 100:.2f}%",
#                                 f"{(int(np.sum(pred_rf)) / total_transactions) * 100:.2f}%",
#                                 f"{(int(np.sum(pred_xgb)) / total_transactions) * 100:.2f}%",
#                                 f"{(int(np.sum(pred_lr)) / total_transactions) * 100:.2f}%"
#                             ],
#                             'Avg Probability': [
#                                 f"{np.mean(prob_dt):.4f}",
#                                 f"{np.mean(prob_rf):.4f}",
#                                 f"{np.mean(prob_xgb):.4f}",
#                                 f"{np.mean(prob_lr):.4f}"
#                             ]
#                         }
                        
#                         if y_true is not None:
#                             comparison_data['Recall'] = [
#                                 f"{recall_dt:.4f}",
#                                 f"{recall_rf:.4f}",
#                                 f"{recall_xgb:.4f}",
#                                 f"{recall_lr:.4f}"
#                             ]
#                             comparison_data['ROC AUC'] = [
#                                 f"{roc_auc_dt:.4f}",
#                                 f"{roc_auc_rf:.4f}",
#                                 f"{roc_auc_xgb:.4f}",
#                                 f"{roc_auc_lr:.4f}"
#                             ]
                        
#                         comparison_df = pd.DataFrame(comparison_data)
#                         st.dataframe(comparison_df, use_container_width=True, hide_index=True)
                        
#                         st.divider()
                        
#                         st.subheader("Visualizations")
#                         col1, col2 = st.columns(2)
                        
#                         with col1:
#                             fig1, axes = plt.subplots(2, 2, figsize=(14, 10))
#                             fig1.patch.set_facecolor('white')
                            
#                             models_data = [
#                                 ('Decision Tree', int(np.sum(pred_dt)), total_transactions - int(np.sum(pred_dt)), '#2ecc71'),
#                                 ('Random Forest', int(np.sum(pred_rf)), total_transactions - int(np.sum(pred_rf)), '#6a82fb'),
#                                 ('XGBoost', int(np.sum(pred_xgb)), total_transactions - int(np.sum(pred_xgb)), '#ffb800'),
#                                 ('Logistic Regression', int(np.sum(pred_lr)), total_transactions - int(np.sum(pred_lr)), '#fc5c7d')
#                             ]
                            
#                             for idx, (model_name, fraud_count, legit_count, color) in enumerate(models_data):
#                                 row = idx // 2
#                                 col = idx % 2
#                                 axes[row, col].pie([fraud_count, legit_count], labels=['Fraudulent', 'Legitimate'], 
#                                              autopct='%1.1f%%', colors=[color, '#95a5a6'], startangle=90, 
#                                              textprops={'color':'#232946','fontweight':'bold'})
#                                 axes[row, col].set_title(model_name, fontsize=12, fontweight='bold')
                            
#                             plt.tight_layout()
#                             st.pyplot(fig1)
                        
#                         with col2:
#                             fig2, ax2 = plt.subplots(figsize=(11, 10))
#                             fig2.patch.set_facecolor('white')
                            
#                             ax2.hist(prob_dt, bins=25, alpha=0.5, label='Decision Tree', color='#2ecc71', edgecolor='black')
#                             ax2.hist(prob_rf, bins=25, alpha=0.5, label='Random Forest', color='#6a82fb', edgecolor='black')
#                             ax2.hist(prob_xgb, bins=25, alpha=0.5, label='XGBoost', color='#ffb800', edgecolor='black')
#                             ax2.hist(prob_lr, bins=25, alpha=0.5, label='Logistic Regression', color='#fc5c7d', edgecolor='black')
                            
#                             ax2.set_xlabel('Fraud Probability', fontsize=11, fontweight='bold')
#                             ax2.set_ylabel('Count', fontsize=11, fontweight='bold')
#                             ax2.set_title('Probability Distribution - Model Comparison', fontsize=12, fontweight='bold')
#                             ax2.legend(fontsize=10)
#                             ax2.grid(True, alpha=0.3)
                            
#                             st.pyplot(fig2)
                        
#                         st.divider()
                        
#                         models_count = (pred_dt.astype(int) + pred_rf.astype(int) + pred_xgb.astype(int) + pred_lr.astype(int))
#                         avg_prob = (prob_dt + prob_rf + prob_xgb + prob_lr) / 4
                        
#                         fraud_4_models = np.where(models_count == 4)[0]
#                         fraud_3_models = np.where(models_count == 3)[0]
#                         fraud_2_models = np.where(models_count == 2)[0]
#                         fraud_1_model = np.where(models_count == 1)[0]
                        
#                         st.subheader("100% FRAUD - Detected by All 4 Models")
#                         if len(fraud_4_models) > 0:
#                             fraud_4_df = df.iloc[fraud_4_models].copy()
#                             fraud_4_df['DT Prob'] = prob_dt[fraud_4_models]
#                             fraud_4_df['RF Prob'] = prob_rf[fraud_4_models]
#                             fraud_4_df['XGB Prob'] = prob_xgb[fraud_4_models]
#                             fraud_4_df['LR Prob'] = prob_lr[fraud_4_models]
#                             fraud_4_df['Avg Probability'] = avg_prob[fraud_4_models]
                            
#                             display_cols = ['Amount', 'DT Prob', 'RF Prob', 'XGB Prob', 'LR Prob', 'Avg Probability']
#                             st.dataframe(fraud_4_df[display_cols].sort_values('Avg Probability', ascending=False).round(4), 
#                                         use_container_width=True, hide_index=True)
#                             st.success(f"Found {len(fraud_4_models)} HIGHEST CONFIDENCE fraud transactions!")
#                         else:
#                             st.info("No transactions detected as fraud by all 4 models.")
                        
#                         st.divider()
                        
#                         st.subheader("HIGH RISK - Detected by 3 Models")
#                         if len(fraud_3_models) > 0:
#                             fraud_3_df = df.iloc[fraud_3_models].copy()
#                             fraud_3_df['DT Prob'] = prob_dt[fraud_3_models]
#                             fraud_3_df['RF Prob'] = prob_rf[fraud_3_models]
#                             fraud_3_df['XGB Prob'] = prob_xgb[fraud_3_models]
#                             fraud_3_df['LR Prob'] = prob_lr[fraud_3_models]
#                             fraud_3_df['Avg Probability'] = avg_prob[fraud_3_models]
#                             fraud_3_df['Models Flagged'] = models_count[fraud_3_models]
                            
#                             display_cols = ['Amount', 'DT Prob', 'RF Prob', 'XGB Prob', 'LR Prob', 'Avg Probability', 'Models Flagged']
#                             st.dataframe(fraud_3_df[display_cols].sort_values('Avg Probability', ascending=False).round(4), 
#                                         use_container_width=True, hide_index=True)
#                             st.warning(f"Found {len(fraud_3_models)} HIGH RISK fraud transactions!")
#                         else:
#                             st.info("No transactions detected as fraud by exactly 3 models.")
                        
#                         st.divider()
                        
#                         st.subheader("MEDIUM RISK - Detected by 2 Models")
#                         if len(fraud_2_models) > 0:
#                             fraud_2_df = df.iloc[fraud_2_models].copy()
#                             fraud_2_df['DT Prob'] = prob_dt[fraud_2_models]
#                             fraud_2_df['RF Prob'] = prob_rf[fraud_2_models]
#                             fraud_2_df['XGB Prob'] = prob_xgb[fraud_2_models]
#                             fraud_2_df['LR Prob'] = prob_lr[fraud_2_models]
#                             fraud_2_df['Avg Probability'] = avg_prob[fraud_2_models]
#                             fraud_2_df['Models Flagged'] = models_count[fraud_2_models]
                            
#                             display_cols = ['Amount', 'DT Prob', 'RF Prob', 'XGB Prob', 'LR Prob', 'Avg Probability', 'Models Flagged']
#                             st.dataframe(fraud_2_df[display_cols].sort_values('Avg Probability', ascending=False).round(4), 
#                                         use_container_width=True, hide_index=True)
#                             st.info(f"Found {len(fraud_2_models)} MEDIUM RISK fraud transactions!")
#                         else:
#                             st.info("No transactions detected as fraud by exactly 2 models.")
                        
#                         st.divider()
                        
#                         st.subheader("LOW RISK - Detected by 1 Model")
#                         if len(fraud_1_model) > 0:
#                             fraud_1_df = df.iloc[fraud_1_model].copy()
#                             fraud_1_df['DT Prob'] = prob_dt[fraud_1_model]
#                             fraud_1_df['RF Prob'] = prob_rf[fraud_1_model]
#                             fraud_1_df['XGB Prob'] = prob_xgb[fraud_1_model]
#                             fraud_1_df['LR Prob'] = prob_lr[fraud_1_model]
#                             fraud_1_df['Avg Probability'] = avg_prob[fraud_1_model]
#                             fraud_1_df['Models Flagged'] = models_count[fraud_1_model]
                            
#                             display_cols = ['Amount', 'DT Prob', 'RF Prob', 'XGB Prob', 'LR Prob', 'Avg Probability', 'Models Flagged']
#                             st.dataframe(fraud_1_df[display_cols].sort_values('Avg Probability', ascending=False).round(4), 
#                                         use_container_width=True, hide_index=True)
#                             st.info(f"Found {len(fraud_1_model)} LOW RISK fraud transactions!")
#                         else:
#                             st.info("No transactions detected as fraud by exactly 1 model.")
                        
#                         st.divider()
                        
#                         st.subheader("Risk Summary")
#                         col1, col2, col3, col4 = st.columns(4)
                        
#                         with col1:
#                             st.metric("100% Fraud (4/4)", len(fraud_4_models))
                        
#                         with col2:
#                             st.metric("High Risk (3/4)", len(fraud_3_models))
                        
#                         with col3:
#                             st.metric("Medium Risk (2/4)", len(fraud_2_models))
                        
#                         with col4:
#                             st.metric("Low Risk (1/4)", len(fraud_1_model))
                        
#         except Exception as e:
#             st.error(f"Processing error: {str(e)}")

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import PyPDF2
import re
import time
from datetime import datetime, timedelta
from sklearn.metrics import roc_curve, auc, recall_score

# Define allowed extensions
ALLOWED_EXTENSIONS = ['csv', 'pdf']
DECISION_TREE_PATH = 'decision_tree_model.joblib'
RANDOM_FOREST_PATH = 'random_Forest_model.pkl'
XGBOOST_PATH = 'XGBoost_model.joblib'
LOGISTIC_REGRESSION_PATH = 'logistic_regression_model.joblib'
SCALER_PATH = 'scaler.pkl'
SCALER_JOBLIB_PATH = 'scaler.joblib'

@st.cache_resource
def load_models():
    try:
        dt_model = joblib.load(DECISION_TREE_PATH)
        rf_model = joblib.load(RANDOM_FOREST_PATH)
        xgb_model = joblib.load(XGBOOST_PATH)
        lr_model = joblib.load(LOGISTIC_REGRESSION_PATH)
        
        # Try to load scaler.pkl first, if not found try scaler.joblib
        try:
            scaler = joblib.load(SCALER_PATH)
        except:
            scaler = joblib.load(SCALER_JOBLIB_PATH)
        
        return dt_model, rf_model, xgb_model, lr_model, scaler
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None, None, None

dt_model, rf_model, xgb_model, lr_model, scaler = load_models()

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
                return None, None, f"Missing required columns: {', '.join(missing_cols)}"
        for col in required_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.dropna(subset=required_cols)
        if len(df) == 0:
            return None, None, "No valid data found after cleaning."
        
        # Check if Class column exists for true labels
        y_true = None
        if 'Class' in df.columns:
            df['Class'] = pd.to_numeric(df['Class'], errors='coerce')
            y_true = df['Class'].values
        
        feature_cols = required_cols
        X = df[feature_cols].copy()
        X_scaled = scaler.transform(X)
        return X_scaled, y_true, None
    except Exception as e:
        return None, None, f"Preprocessing error: {str(e)}"

def predict_fraud(X_scaled, model_choice):
    try:
        if model_choice == "dt":
            model = dt_model
        elif model_choice == "rf":
            model = rf_model
        elif model_choice == "xgb":
            model = xgb_model
        elif model_choice == "lr":
            model = lr_model
        else:
            st.error("Unknown model selected.")
            return None, None
        predictions = model.predict(X_scaled)
        probabilities = model.predict_proba(X_scaled)[:, 1]
        return predictions, probabilities
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None, None

# Generate demo transaction data
def generate_demo_transaction():
    transaction_id = f"TXN{np.random.randint(100000, 999999)}"
    amount = np.random.choice([
        np.random.uniform(5, 500),  # Normal transactions
        np.random.uniform(1000, 5000) if np.random.random() < 0.15 else np.random.uniform(5, 500)  # Some high amounts
    ])
    
    # Simulate fraud probability (15% chance of fraud)
    is_fraud = np.random.random() < 0.15
    fraud_prob = np.random.uniform(0.75, 0.99) if is_fraud else np.random.uniform(0.01, 0.35)
    
    merchant = np.random.choice([
        "Amazon", "Walmart", "Target", "Starbucks", "McDonald's", 
        "Shell Gas", "Apple Store", "Best Buy", "Netflix", "Uber"
    ])
    
    card_last4 = f"****{np.random.randint(1000, 9999)}"
    
    return {
        'id': transaction_id,
        'timestamp': datetime.now(),
        'amount': amount,
        'merchant': merchant,
        'card': card_last4,
        'fraud_prob': fraud_prob,
        'is_fraud': is_fraud,
        'status': 'pending',
        'created_at': time.time()
    }

# Initialize session state for demo dashboard
if 'demo_transactions' not in st.session_state:
    st.session_state.demo_transactions = []
    st.session_state.demo_stats = {
        'total': 0,
        'fraud_detected': 0,
        'fraud_blocked': 0,
        'legitimate': 0,
        'under_review': 0
    }
    st.session_state.last_update = time.time()

# Initialize session state for analysis results (to preserve them during auto-refresh)
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None

st.set_page_config(page_title="Fraud Detection", layout="wide")

st.title("🔒 Fraud Detection System")
st.markdown("### CREDIT CARD TRANSACTION ANALYSIS")

# Create two columns - main content and live dashboard
main_col, demo_col = st.columns([2, 1])

with demo_col:
    st.markdown("### 🎯 Live Demo Dashboard")
    st.markdown("*Future Implementation Preview*")
    
    # Create placeholders for auto-updating content
    stats_placeholder = st.empty()
    feed_placeholder = st.empty()
    
    # Auto-update logic
    current_time = time.time()
    if current_time - st.session_state.last_update > 3:  # Add new transaction every 3 seconds
        new_txn = generate_demo_transaction()
        st.session_state.demo_transactions.insert(0, new_txn)
        
        # Keep only last 50 transactions
        if len(st.session_state.demo_transactions) > 50:
            st.session_state.demo_transactions = st.session_state.demo_transactions[:50]
        
        st.session_state.last_update = current_time
    
    # Auto-update transaction statuses
    for txn in st.session_state.demo_transactions:
        age = time.time() - txn['created_at']
        
        if txn['status'] == 'pending' and age > 2:
            if txn['fraud_prob'] > 0.7:
                txn['status'] = 'reviewing'
            elif txn['is_fraud']:
                txn['status'] = 'reviewing'
            else:
                txn['status'] = 'approved'
        
        if txn['status'] == 'reviewing' and age > 5:
            if txn['is_fraud']:
                txn['status'] = 'blocked'
            else:
                txn['status'] = 'approved'
    
    # Update stats
    st.session_state.demo_stats['total'] = len(st.session_state.demo_transactions)
    st.session_state.demo_stats['fraud_detected'] = sum(1 for t in st.session_state.demo_transactions if t['is_fraud'])
    st.session_state.demo_stats['fraud_blocked'] = sum(1 for t in st.session_state.demo_transactions if t['is_fraud'] and t['status'] == 'blocked')
    st.session_state.demo_stats['legitimate'] = sum(1 for t in st.session_state.demo_transactions if not t['is_fraud'])
    st.session_state.demo_stats['under_review'] = sum(1 for t in st.session_state.demo_transactions if t['status'] == 'reviewing')
    
    # Display stats in placeholder
    with stats_placeholder.container():
        st.markdown("#### 📊 Real-time Statistics")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total", st.session_state.demo_stats['total'])
            st.metric("🔴 Fraud", st.session_state.demo_stats['fraud_detected'])
        with col2:
            st.metric("✅ Legitimate", st.session_state.demo_stats['legitimate'])
            st.metric("🛡️ Blocked", st.session_state.demo_stats['fraud_blocked'])
        
        st.markdown("---")
        st.markdown("#### 🔄 Live Transaction Feed")
    
    # Display transactions in placeholder
    with feed_placeholder.container():
        for txn in st.session_state.demo_transactions[:20]:  # Show last 20
            age = time.time() - txn['created_at']
            
            # Color coding based on status
            if txn['status'] == 'blocked':
                border_color = '#e74c3c'
                bg_color = '#fadbd8'
                status_emoji = '🚫'
                status_text = 'BLOCKED'
            elif txn['status'] == 'reviewing':
                border_color = '#f39c12'
                bg_color = '#fef5e7'
                status_emoji = '⚠️'
                status_text = 'REVIEWING'
            elif txn['is_fraud'] and txn['status'] == 'pending':
                border_color = '#e67e22'
                bg_color = '#fef5e7'
                status_emoji = '🔍'
                status_text = 'DETECTING'
            elif txn['status'] == 'approved':
                border_color = '#27ae60'
                bg_color = '#d5f4e6'
                status_emoji = '✅'
                status_text = 'APPROVED'
            else:
                border_color = '#3498db'
                bg_color = '#ebf5fb'
                status_emoji = '⏳'
                status_text = 'PENDING'
            
            st.markdown(f"""
            <div style="border-left: 4px solid {border_color}; background-color: {bg_color}; padding: 10px; margin-bottom: 10px; border-radius: 5px;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <strong style="font-size: 14px;">{txn['merchant']}</strong><br>
                        <span style="font-size: 12px; color: #555;">{txn['card']} • ${txn['amount']:.2f}</span><br>
                        <span style="font-size: 11px; color: #888;">{int(age)}s ago</span>
                    </div>
                    <div style="text-align: right;">
                        <span style="font-size: 18px;">{status_emoji}</span><br>
                        <strong style="font-size: 11px; color: {border_color};">{status_text}</strong><br>
                        <span style="font-size: 10px; color: #666;">Risk: {txn['fraud_prob']:.0%}</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("*🔄 Auto-refreshes every 3 seconds*")
        st.markdown("*🤖 Simulated AI fraud detection in action*")
    
    # Auto-refresh only the demo dashboard every 3 seconds
    time.sleep(3)
    st.rerun()

with main_col:
    st.info("How to Use:\n- Upload your credit card transaction data (CSV or PDF)\n- File must have columns: Time, V1-V28, Amount\n- Optional: Include 'Class' column (0=Legitimate, 1=Fraud) for ROC AUC and Recall metrics\n- Click Check Transaction to analyze with all 4 models\n- Get instant fraud detection results with model comparison!")

    uploaded_file = st.file_uploader("Choose your file (CSV or PDF)", type=ALLOWED_EXTENSIONS)
    check = st.button("Check Transaction", use_container_width=True)

    # Display previous results if they exist
    if st.session_state.analysis_results is not None and not check:
        results = st.session_state.analysis_results
        
        st.subheader("Overall Summary")
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Total Transactions", results['total_transactions'])
        
        with col2:
            st.metric("Decision Tree Fraud", results['dt_fraud'])
        
        with col3:
            st.metric("Random Forest Fraud", results['rf_fraud'])
        
        with col4:
            st.metric("XGBoost Fraud", results['xgb_fraud'])
        
        with col5:
            st.metric("Logistic Regression Fraud", results['lr_fraud'])
        
        st.divider()
        
        # Display metrics if available
        if results['has_true_labels']:
            st.subheader("Model Performance Metrics")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.pyplot(results['roc_fig'])
            
            with col2:
                st.pyplot(results['recall_fig'])
            
            st.divider()
        
        st.subheader("Model Comparison")
        st.dataframe(results['comparison_df'], use_container_width=True, hide_index=True)
        
        st.divider()
        
        st.subheader("Visualizations")
        col1, col2 = st.columns(2)
        
        with col1:
            st.pyplot(results['pie_fig'])
        
        with col2:
            st.pyplot(results['hist_fig'])
        
        st.divider()
        
        st.subheader("100% FRAUD - Detected by All 4 Models")
        if results['fraud_4_count'] > 0:
            st.dataframe(results['fraud_4_df'], use_container_width=True, hide_index=True)
            st.success(f"Found {results['fraud_4_count']} HIGHEST CONFIDENCE fraud transactions!")
        else:
            st.info("No transactions detected as fraud by all 4 models.")
        
        st.divider()
        
        st.subheader("HIGH RISK - Detected by 3 Models")
        if results['fraud_3_count'] > 0:
            st.dataframe(results['fraud_3_df'], use_container_width=True, hide_index=True)
            st.warning(f"Found {results['fraud_3_count']} HIGH RISK fraud transactions!")
        else:
            st.info("No transactions detected as fraud by exactly 3 models.")
        
        st.divider()
        
        st.subheader("MEDIUM RISK - Detected by 2 Models")
        if results['fraud_2_count'] > 0:
            st.dataframe(results['fraud_2_df'], use_container_width=True, hide_index=True)
            st.info(f"Found {results['fraud_2_count']} MEDIUM RISK fraud transactions!")
        else:
            st.info("No transactions detected as fraud by exactly 2 models.")
        
        st.divider()
        
        st.subheader("LOW RISK - Detected by 1 Model")
        if results['fraud_1_count'] > 0:
            st.dataframe(results['fraud_1_df'], use_container_width=True, hide_index=True)
            st.info(f"Found {results['fraud_1_count']} LOW RISK fraud transactions!")
        else:
            st.info("No transactions detected as fraud by exactly 1 model.")
        
        st.divider()
        
        st.subheader("Risk Summary")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("100% Fraud (4/4)", results['fraud_4_count'])
        
        with col2:
            st.metric("High Risk (3/4)", results['fraud_3_count'])
        
        with col3:
            st.metric("Medium Risk (2/4)", results['fraud_2_count'])
        
        with col4:
            st.metric("Low Risk (1/4)", results['fraud_1_count'])

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
                    X_scaled, y_true, error = preprocess_data(df)
                    if error:
                        st.error(error)
                    else:
                        pred_dt, prob_dt = predict_fraud(X_scaled, "dt")
                        pred_rf, prob_rf = predict_fraud(X_scaled, "rf")
                        pred_xgb, prob_xgb = predict_fraud(X_scaled, "xgb")
                        pred_lr, prob_lr = predict_fraud(X_scaled, "lr")
                        
                        if pred_dt is None or pred_rf is None or pred_xgb is None or pred_lr is None:
                            st.error("Prediction failed.")
                        else:
                            total_transactions = len(pred_dt)
                            
                            # Prepare all results to store in session state
                            results = {
                                'total_transactions': total_transactions,
                                'dt_fraud': int(np.sum(pred_dt)),
                                'rf_fraud': int(np.sum(pred_rf)),
                                'xgb_fraud': int(np.sum(pred_xgb)),
                                'lr_fraud': int(np.sum(pred_lr)),
                                'has_true_labels': y_true is not None
                            }
                            
                            # Calculate metrics if true labels are available
                            if y_true is not None:
                                recall_dt = recall_score(y_true, pred_dt)
                                recall_rf = recall_score(y_true, pred_rf)
                                recall_xgb = recall_score(y_true, pred_xgb)
                                recall_lr = recall_score(y_true, pred_lr)
                                
                                fpr_dt, tpr_dt, _ = roc_curve(y_true, prob_dt)
                                fpr_rf, tpr_rf, _ = roc_curve(y_true, prob_rf)
                                fpr_xgb, tpr_xgb, _ = roc_curve(y_true, prob_xgb)
                                fpr_lr, tpr_lr, _ = roc_curve(y_true, prob_lr)
                                
                                roc_auc_dt = auc(fpr_dt, tpr_dt)
                                roc_auc_rf = auc(fpr_rf, tpr_rf)
                                roc_auc_xgb = auc(fpr_xgb, tpr_xgb)
                                roc_auc_lr = auc(fpr_lr, tpr_lr)
                                
                                # Create ROC figure
                                fig_roc, ax_roc = plt.subplots(figsize=(10, 8))
                                fig_roc.patch.set_facecolor('white')
                                
                                ax_roc.plot(fpr_dt, tpr_dt, color='#2ecc71', lw=2, 
                                           label=f'Decision Tree (AUC = {roc_auc_dt:.4f})')
                                ax_roc.plot(fpr_rf, tpr_rf, color='#6a82fb', lw=2, 
                                           label=f'Random Forest (AUC = {roc_auc_rf:.4f})')
                                ax_roc.plot(fpr_xgb, tpr_xgb, color='#ffb800', lw=2, 
                                           label=f'XGBoost (AUC = {roc_auc_xgb:.4f})')
                                ax_roc.plot(fpr_lr, tpr_lr, color='#fc5c7d', lw=2, 
                                           label=f'Logistic Regression (AUC = {roc_auc_lr:.4f})')
                                ax_roc.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', label='Random Classifier')
                                
                                ax_roc.set_xlim([0.0, 1.0])
                                ax_roc.set_ylim([0.0, 1.05])
                                ax_roc.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
                                ax_roc.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
                                ax_roc.set_title('ROC Curve - Model Comparison', fontsize=14, fontweight='bold')
                                ax_roc.legend(loc="lower right", fontsize=9)
                                ax_roc.grid(True, alpha=0.3)
                                
                                results['roc_fig'] = fig_roc
                                
                                # Create Recall figure
                                fig_recall, ax_recall = plt.subplots(figsize=(10, 8))
                                fig_recall.patch.set_facecolor('white')
                                
                                models = ['Decision Tree', 'Random Forest', 'XGBoost', 'Logistic Reg']
                                recalls = [recall_dt, recall_rf, recall_xgb, recall_lr]
                                colors = ['#2ecc71', '#6a82fb', '#ffb800', '#fc5c7d']
                                
                                bars = ax_recall.bar(models, recalls, color=colors, edgecolor='black', linewidth=1.5)
                                
                                for bar, recall in zip(bars, recalls):
                                    height = bar.get_height()
                                    ax_recall.text(bar.get_x() + bar.get_width()/2., height,
                                                  f'{recall:.4f}',
                                                  ha='center', va='bottom', fontsize=11, fontweight='bold')
                                
                                ax_recall.set_ylim([0, 1.1])
                                ax_recall.set_ylabel('Recall Score', fontsize=12, fontweight='bold')
                                ax_recall.set_title('Recall Score - Model Comparison', fontsize=14, fontweight='bold')
                                ax_recall.grid(True, axis='y', alpha=0.3)
                                plt.xticks(rotation=15, ha='right')
                                
                                results['recall_fig'] = fig_recall
                            
                            # Model comparison data
                            comparison_data = {
                                'Model': ['Decision Tree', 'Random Forest', 'XGBoost', 'Logistic Regression'],
                                'Fraudulent': [
                                    int(np.sum(pred_dt)),
                                    int(np.sum(pred_rf)),
                                    int(np.sum(pred_xgb)),
                                    int(np.sum(pred_lr))
                                ],
                                'Legitimate': [
                                    total_transactions - int(np.sum(pred_dt)),
                                    total_transactions - int(np.sum(pred_rf)),
                                    total_transactions - int(np.sum(pred_xgb)),
                                    total_transactions - int(np.sum(pred_lr))
                                ],
                                'Fraud %': [
                                    f"{(int(np.sum(pred_dt)) / total_transactions) * 100:.2f}%",
                                    f"{(int(np.sum(pred_rf)) / total_transactions) * 100:.2f}%",
                                    f"{(int(np.sum(pred_xgb)) / total_transactions) * 100:.2f}%",
                                    f"{(int(np.sum(pred_lr)) / total_transactions) * 100:.2f}%"
                                ],
                                'Avg Probability': [
                                    f"{np.mean(prob_dt):.4f}",
                                    f"{np.mean(prob_rf):.4f}",
                                    f"{np.mean(prob_xgb):.4f}",
                                    f"{np.mean(prob_lr):.4f}"
                                ]
                            }
                            
                            if y_true is not None:
                                comparison_data['Recall'] = [
                                    f"{recall_dt:.4f}",
                                    f"{recall_rf:.4f}",
                                    f"{recall_xgb:.4f}",
                                    f"{recall_lr:.4f}"
                                ]
                                comparison_data['ROC AUC'] = [
                                    f"{roc_auc_dt:.4f}",
                                    f"{roc_auc_rf:.4f}",
                                    f"{roc_auc_xgb:.4f}",
                                    f"{roc_auc_lr:.4f}"
                                ]
                            
                            results['comparison_df'] = pd.DataFrame(comparison_data)
                            
                            # Create pie charts
                            fig1, axes = plt.subplots(2, 2, figsize=(14, 10))
                            fig1.patch.set_facecolor('white')
                            
                            models_data = [
                                ('Decision Tree', int(np.sum(pred_dt)), total_transactions - int(np.sum(pred_dt)), '#2ecc71'),
                                ('Random Forest', int(np.sum(pred_rf)), total_transactions - int(np.sum(pred_rf)), '#6a82fb'),
                                ('XGBoost', int(np.sum(pred_xgb)), total_transactions - int(np.sum(pred_xgb)), '#ffb800'),
                                ('Logistic Regression', int(np.sum(pred_lr)), total_transactions - int(np.sum(pred_lr)), '#fc5c7d')
                            ]
                            
                            for idx, (model_name, fraud_count, legit_count, color) in enumerate(models_data):
                                row = idx // 2
                                col = idx % 2
                                axes[row, col].pie([fraud_count, legit_count], labels=['Fraudulent', 'Legitimate'], 
                                             autopct='%1.1f%%', colors=[color, '#95a5a6'], startangle=90, 
                                             textprops={'color':'#232946','fontweight':'bold'})
                                axes[row, col].set_title(model_name, fontsize=12, fontweight='bold')
                            
                            plt.tight_layout()
                            results['pie_fig'] = fig1
                            
                            # Create histogram
                            fig2, ax2 = plt.subplots(figsize=(11, 10))
                            fig2.patch.set_facecolor('white')
                            
                            ax2.hist(prob_dt, bins=25, alpha=0.5, label='Decision Tree', color='#2ecc71', edgecolor='black')
                            ax2.hist(prob_rf, bins=25, alpha=0.5, label='Random Forest', color='#6a82fb', edgecolor='black')
                            ax2.hist(prob_xgb, bins=25, alpha=0.5, label='XGBoost', color='#ffb800', edgecolor='black')
                            ax2.hist(prob_lr, bins=25, alpha=0.5, label='Logistic Regression', color='#fc5c7d', edgecolor='black')
                            
                            ax2.set_xlabel('Fraud Probability', fontsize=11, fontweight='bold')
                            ax2.set_ylabel('Count', fontsize=11, fontweight='bold')
                            ax2.set_title('Probability Distribution - Model Comparison', fontsize=12, fontweight='bold')
                            ax2.legend(fontsize=10)
                            ax2.grid(True, alpha=0.3)
                            
                            results['hist_fig'] = fig2
                            
                            # Calculate risk categories
                            models_count = (pred_dt.astype(int) + pred_rf.astype(int) + pred_xgb.astype(int) + pred_lr.astype(int))
                            avg_prob = (prob_dt + prob_rf + prob_xgb + prob_lr) / 4
                            
                            fraud_4_models = np.where(models_count == 4)[0]
                            fraud_3_models = np.where(models_count == 3)[0]
                            fraud_2_models = np.where(models_count == 2)[0]
                            fraud_1_model = np.where(models_count == 1)[0]
                            
                            # Prepare fraud dataframes
                            if len(fraud_4_models) > 0:
                                fraud_4_df = df.iloc[fraud_4_models].copy()
                                fraud_4_df['DT Prob'] = prob_dt[fraud_4_models]
                                fraud_4_df['RF Prob'] = prob_rf[fraud_4_models]
                                fraud_4_df['XGB Prob'] = prob_xgb[fraud_4_models]
                                fraud_4_df['LR Prob'] = prob_lr[fraud_4_models]
                                fraud_4_df['Avg Probability'] = avg_prob[fraud_4_models]
                                display_cols = ['Amount', 'DT Prob', 'RF Prob', 'XGB Prob', 'LR Prob', 'Avg Probability']
                                results['fraud_4_df'] = fraud_4_df[display_cols].sort_values('Avg Probability', ascending=False).round(4)
                                results['fraud_4_count'] = len(fraud_4_models)
                            else:
                                results['fraud_4_df'] = None
                                results['fraud_4_count'] = 0
                            
                            if len(fraud_3_models) > 0:
                                fraud_3_df = df.iloc[fraud_3_models].copy()
                                fraud_3_df['DT Prob'] = prob_dt[fraud_3_models]
                                fraud_3_df['RF Prob'] = prob_rf[fraud_3_models]
                                fraud_3_df['XGB Prob'] = prob_xgb[fraud_3_models]
                                fraud_3_df['LR Prob'] = prob_lr[fraud_3_models]
                                fraud_3_df['Avg Probability'] = avg_prob[fraud_3_models]
                                fraud_3_df['Models Flagged'] = models_count[fraud_3_models]
                                display_cols = ['Amount', 'DT Prob', 'RF Prob', 'XGB Prob', 'LR Prob', 'Avg Probability', 'Models Flagged']
                                results['fraud_3_df'] = fraud_3_df[display_cols].sort_values('Avg Probability', ascending=False).round(4)
                                results['fraud_3_count'] = len(fraud_3_models)
                            else:
                                results['fraud_3_df'] = None
                                results['fraud_3_count'] = 0
                            
                            if len(fraud_2_models) > 0:
                                fraud_2_df = df.iloc[fraud_2_models].copy()
                                fraud_2_df['DT Prob'] = prob_dt[fraud_2_models]
                                fraud_2_df['RF Prob'] = prob_rf[fraud_2_models]
                                fraud_2_df['XGB Prob'] = prob_xgb[fraud_2_models]
                                fraud_2_df['LR Prob'] = prob_lr[fraud_2_models]
                                fraud_2_df['Avg Probability'] = avg_prob[fraud_2_models]
                                fraud_2_df['Models Flagged'] = models_count[fraud_2_models]
                                display_cols = ['Amount', 'DT Prob', 'RF Prob', 'XGB Prob', 'LR Prob', 'Avg Probability', 'Models Flagged']
                                results['fraud_2_df'] = fraud_2_df[display_cols].sort_values('Avg Probability', ascending=False).round(4)
                                results['fraud_2_count'] = len(fraud_2_models)
                            else:
                                results['fraud_2_df'] = None
                                results['fraud_2_count'] = 0
                            
                            if len(fraud_1_model) > 0:
                                fraud_1_df = df.iloc[fraud_1_model].copy()
                                fraud_1_df['DT Prob'] = prob_dt[fraud_1_model]
                                fraud_1_df['RF Prob'] = prob_rf[fraud_1_model]
                                fraud_1_df['XGB Prob'] = prob_xgb[fraud_1_model]
                                fraud_1_df['LR Prob'] = prob_lr[fraud_1_model]
                                fraud_1_df['Avg Probability'] = avg_prob[fraud_1_model]
                                fraud_1_df['Models Flagged'] = models_count[fraud_1_model]
                                display_cols = ['Amount', 'DT Prob', 'RF Prob', 'XGB Prob', 'LR Prob', 'Avg Probability', 'Models Flagged']
                                results['fraud_1_df'] = fraud_1_df[display_cols].sort_values('Avg Probability', ascending=False).round(4)
                                results['fraud_1_count'] = len(fraud_1_model)
                            else:
                                results['fraud_1_df'] = None
                                results['fraud_1_count'] = 0
                            
                            # Store results in session state
                            st.session_state.analysis_results = results
                            
                            # Display results immediately
                            st.subheader("Overall Summary")
                            col1, col2, col3, col4, col5 = st.columns(5)
                            
                            with col1:
                                st.metric("Total Transactions", total_transactions)
                            
                            with col2:
                                st.metric("Decision Tree Fraud", int(np.sum(pred_dt)))
                            
                            with col3:
                                st.metric("Random Forest Fraud", int(np.sum(pred_rf)))
                            
                            with col4:
                                st.metric("XGBoost Fraud", int(np.sum(pred_xgb)))
                            
                            with col5:
                                st.metric("Logistic Regression Fraud", int(np.sum(pred_lr)))
                            
                            st.divider()
                            
                            if y_true is not None:
                                st.subheader("Model Performance Metrics")
                                
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    st.pyplot(results['roc_fig'])
                                
                                with col2:
                                    st.pyplot(results['recall_fig'])
                                
                                st.divider()
                            
                            st.subheader("Model Comparison")
                            st.dataframe(results['comparison_df'], use_container_width=True, hide_index=True)
                            
                            st.divider()
                            
                            st.subheader("Visualizations")
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.pyplot(results['pie_fig'])
                            
                            with col2:
                                st.pyplot(results['hist_fig'])
                            
                            st.divider()
                            
                            st.subheader("100% FRAUD - Detected by All 4 Models")
                            if results['fraud_4_count'] > 0:
                                st.dataframe(results['fraud_4_df'], use_container_width=True, hide_index=True)
                                st.success(f"Found {results['fraud_4_count']} HIGHEST CONFIDENCE fraud transactions!")
                            else:
                                st.info("No transactions detected as fraud by all 4 models.")
                            
                            st.divider()
                            
                            st.subheader("HIGH RISK - Detected by 3 Models")
                            if results['fraud_3_count'] > 0:
                                st.dataframe(results['fraud_3_df'], use_container_width=True, hide_index=True)
                                st.warning(f"Found {results['fraud_3_count']} HIGH RISK fraud transactions!")
                            else:
                                st.info("No transactions detected as fraud by exactly 3 models.")
                            
                            st.divider()
                            
                            st.subheader("MEDIUM RISK - Detected by 2 Models")
                            if results['fraud_2_count'] > 0:
                                st.dataframe(results['fraud_2_df'], use_container_width=True, hide_index=True)
                                st.info(f"Found {results['fraud_2_count']} MEDIUM RISK fraud transactions!")
                            else:
                                st.info("No transactions detected as fraud by exactly 2 models.")
                            
                            st.divider()
                            
                            st.subheader("LOW RISK - Detected by 1 Model")
                            if results['fraud_1_count'] > 0:
                                st.dataframe(results['fraud_1_df'], use_container_width=True, hide_index=True)
                                st.info(f"Found {results['fraud_1_count']} LOW RISK fraud transactions!")
                            else:
                                st.info("No transactions detected as fraud by exactly 1 model.")
                            
                            st.divider()
                            
                            st.subheader("Risk Summary")
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                st.metric("100% Fraud (4/4)", results['fraud_4_count'])
                            
                            with col2:
                                st.metric("High Risk (3/4)", results['fraud_3_count'])
                            
                            with col3:
                                st.metric("Medium Risk (2/4)", results['fraud_2_count'])
                            
                            with col4:
                                st.metric("Low Risk (1/4)", results['fraud_1_count'])
                            
            except Exception as e:
                st.error(f"Processing error: {str(e)}")


