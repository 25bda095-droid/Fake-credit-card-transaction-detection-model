

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
from streamlit_autorefresh import st_autorefresh
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import PyPDF2
import re
from sklearn.metrics import roc_curve, auc, recall_score
import time
from datetime import datetime

# ... [your previous model loading and utility code goes here] ...

if "analysis_results" not in st.session_state:
    st.session_state.analysis_results = None
    st.session_state.show_results = False

with main_col:
    st.title("Fraud Detection System")
    # ... [rest of your upload and check button UI] ...
    if check and uploaded_file is not None:
        # ... [your input/processing code] ...
        if results_ready:
            # Save ALL COMPUTABLE ARRAYS and necessary values
            st.session_state.analysis_results = {
                'total_transactions': total_transactions,
                'names': ["Tuned Model", "Random Forest", "XGBoost"],
                'preds': {key: preds[key].tolist() for key in preds},  # .tolist() for JSON serializable
                'probs': {key: probs[key].tolist() for key in probs},
                'y_true': y_true.tolist() if y_true is not None else None,
                # Save all metrics:
                'recall_scores': recall_scores,
                'roc_aucs': roc_aucs,
            }
            st.session_state.show_results = True

    # ALWAYS display if available
    if st.session_state.show_results and st.session_state.analysis_results:
        res = st.session_state.analysis_results
        total_transactions = res['total_transactions']
        names = res['names']
        preds = {k: np.array(res['preds'][k]) for k in names}
        probs = {k: np.array(res['probs'][k]) for k in names}
        y_true = np.array(res['y_true']) if res['y_true'] is not None else None
        recall_scores = res['recall_scores']
        roc_aucs = res['roc_aucs']

        # -- Regenerate/Show all charts here --
        fig1, axes1 = plt.subplots(1, 3, figsize=(12, 4))
        colors = ['#FC8181', '#6a82fb', '#ffb800']
        for i, name in enumerate(names):
            N_fraud = np.sum(preds[name])
            axes1[i].pie(
                [N_fraud, total_transactions - N_fraud], labels=['Fraudulent', 'Legitimate'],
                autopct='%1.1f%%', colors=[colors[i], '#74b9ff'],
                startangle=90, textprops={'color':'#232946', 'fontweight':'bold'}
            )
            axes1[i].set_title(name)
        st.subheader("Visualizations")
        st.pyplot(fig1)

        fig2, ax2 = plt.subplots(figsize=(8, 4))
        for name, color in zip(names, colors):
            ax2.hist(probs[name], bins=30, alpha=0.5, label=name, edgecolor='black', color=color)
        ax2.set_xlabel('Fraud Probability')
        ax2.set_ylabel('Count')
        ax2.set_title('Probability Distribution - Model Comparison')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        st.pyplot(fig2)

        if y_true is not None:
            fprs = {}
            tprs = {}
            for name in names:
                fprs[name], tprs[name], _ = roc_curve(y_true, probs[name])
            fig3, ax3 = plt.subplots(figsize=(8, 5))
            for name, color in zip(names, colors):
                ax3.plot(fprs[name], tprs[name], color=color, lw=2, label=f"{name} (AUC = {roc_aucs[name]:.4f})")
            ax3.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', label='Random Classifier')
            ax3.set_xlim([0.0, 1.0])
            ax3.set_ylim([0.0, 1.05])
            ax3.set_xlabel('False Positive Rate')
            ax3.set_ylabel('True Positive Rate')
            ax3.set_title('ROC Curve - Model Comparison')
            ax3.legend(loc="lower right")
            st.subheader("Model Performance Metrics")
            st.pyplot(fig3)

            fig4, ax4 = plt.subplots(figsize=(8, 5))
            recall_vals = [recall_scores[n] for n in names]
            bars = ax4.bar(names, recall_vals, color=colors, edgecolor='black', linewidth=1.5)
            for bar, recall in zip(bars, recall_vals):
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height, f'{recall:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
            ax4.set_ylim([0, 1.1])
            ax4.set_ylabel('Recall Score')
            ax4.set_title('Recall Score - Model Comparison')
            ax4.grid(True, axis='y', alpha=0.3)
            plt.xticks(rotation=15, ha='right')
            st.pyplot(fig4)
        # -- Model Comparison Table --
        st.subheader("Model Comparison")
        comp_data = {
            'Model': names,
            'Fraudulent': [np.sum(preds[n]) for n in names],
            'Legitimate': [total_transactions - np.sum(preds[n]) for n in names],
            'Fraud %': [f"{(np.sum(preds[n])/total_transactions)*100:.2f}%" for n in names],
            'Avg Probability': [f"{np.mean(probs[n]):.4f}" for n in names],
            'Recall': [f"{recall_scores[n]:.4f}" for n in names] if y_true is not None else [""]*len(names),
            'ROC AUC': [f"{roc_aucs[n]:.4f}" for n in names] if y_true is not None else [""]*len(names)
        }
        st.dataframe(pd.DataFrame(comp_data), use_container_width=True, hide_index=True)

