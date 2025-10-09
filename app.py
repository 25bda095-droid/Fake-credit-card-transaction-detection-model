from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import joblib
import numpy as np
import os
from werkzeug.utils import secure_filename
import io
import PyPDF2
import re

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'

# Create uploads folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# File paths
MODEL_PATH = r"C:\fraud detection trained models all files\fraud_detection_model_tuned.pkl"
SCALER_PATH = r"C:\fraud detection trained models all files\scaler.pkl"
RANDOM_FOREST_PATH = r"C:\fraud detection trained models all files\random_Forest_model.pkl"

# Load models and scaler
try:
    tuned_model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    rf_model = joblib.load(RANDOM_FOREST_PATH)
    print("✓ Models and scaler loaded successfully!")
except Exception as e:
    print(f"Error loading models: {e}")
    tuned_model = None
    scaler = None
    rf_model = None

ALLOWED_EXTENSIONS = {'csv', 'pdf'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_csv_from_pdf(pdf_file):
    """Extract tabular data from PDF"""
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        
        # Try to parse CSV-like content from PDF
        lines = text.strip().split('\n')
        data = []
        for line in lines:
            # Split by common delimiters
            row = re.split(r'[,\t\s]{2,}', line.strip())
            if len(row) > 1:
                data.append(row)
        
        if data:
            df = pd.DataFrame(data[1:], columns=data[0])
            return df
        return None
    except Exception as e:
        print(f"PDF extraction error: {e}")
        return None

def preprocess_data(df):
    """Preprocess the uploaded data"""
    try:
        # Expected columns: Time, V1-V28, Amount (and possibly Class for labeled data)
        required_cols = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']
        
        # Check if all required columns exist
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            # Try without Time if it's missing
            if 'Time' in missing_cols:
                required_cols = [f'V{i}' for i in range(1, 29)] + ['Amount']
                missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            return None, f"Missing required columns: {', '.join(missing_cols)}"
        
        # Convert to numeric
        for col in required_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Drop rows with NaN values
        df = df.dropna(subset=required_cols)
        
        if len(df) == 0:
            return None, "No valid data found after cleaning"
        
        # Select features for prediction
        if 'Time' in df.columns:
            feature_cols = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']
        else:
            feature_cols = [f'V{i}' for i in range(1, 29)] + ['Amount']
        
        X = df[feature_cols].copy()
        
        # Scale the features
        X_scaled = scaler.transform(X)
        
        return X_scaled, None
    except Exception as e:
        return None, f"Preprocessing error: {str(e)}"

def predict_fraud(X_scaled, model_choice='tuned'):
    """Make fraud predictions"""
    try:
        model = tuned_model if model_choice == 'tuned' else rf_model
        
        # Get predictions
        predictions = model.predict(X_scaled)
        probabilities = model.predict_proba(X_scaled)[:, 1]  # Probability of fraud
        
        return predictions, probabilities
    except Exception as e:
        print(f"Prediction error: {e}")
        return None, None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    model_choice = request.form.get('model', 'tuned')
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file format. Please upload CSV or PDF'}), 400
    
    try:
        # Read the file
        if file.filename.endswith('.csv'):
            df = pd.read_csv(file)
        else:  # PDF
            df = extract_csv_from_pdf(file)
            if df is None:
                return jsonify({'error': 'Could not extract data from PDF'}), 400
        
        # Preprocess data
        X_scaled, error = preprocess_data(df)
        if error:
            return jsonify({'error': error}), 400
        
        # Make predictions
        predictions, probabilities = predict_fraud(X_scaled, model_choice)
        if predictions is None:
            return jsonify({'error': 'Prediction failed'}), 500
        
        # Prepare results
        total_transactions = len(predictions)
        fraud_count = int(np.sum(predictions))
        legitimate_count = total_transactions - fraud_count
        fraud_percentage = (fraud_count / total_transactions) * 100
        
        # Get high-risk transactions (probability > 0.7)
        high_risk_indices = np.where(probabilities > 0.7)[0]
        high_risk_transactions = []
        
        for idx in high_risk_indices[:10]:  # Limit to top 10
            high_risk_transactions.append({
                'index': int(idx),
                'probability': float(probabilities[idx]),
                'amount': float(df.iloc[idx]['Amount']) if 'Amount' in df.columns else 'N/A'
            })
        
        results = {
            'total_transactions': total_transactions,
            'fraud_count': fraud_count,
            'legitimate_count': legitimate_count,
            'fraud_percentage': round(fraud_percentage, 2),
            'high_risk_transactions': high_risk_transactions,
            'model_used': 'Tuned Model' if model_choice == 'tuned' else 'Random Forest Model'
        }
        
        return jsonify(results)
    
    except Exception as e:
        return jsonify({'error': f'Processing error: {str(e)}'}), 500

@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy',
        'models_loaded': tuned_model is not None and scaler is not None
    })

if __name__ == '__main__':
    # Create templates folder and HTML file
    os.makedirs('templates', exist_ok=True)
    
    html_content = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Credit Card Fraud Detection System</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }

        .container {
            max-width: 900px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
            overflow: hidden;
        }

        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }

        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
        }

        .header p {
            font-size: 1.1em;
            opacity: 0.9;
        }

        .content {
            padding: 40px;
        }

        .upload-section {
            background: #f8f9fa;
            border: 3px dashed #667eea;
            border-radius: 15px;
            padding: 40px;
            text-align: center;
            margin-bottom: 30px;
            transition: all 0.3s ease;
        }

        .upload-section:hover {
            border-color: #764ba2;
            background: #f0f0f5;
        }

        .upload-section.dragover {
            background: #e8e9ff;
            border-color: #764ba2;
        }

        .file-input-wrapper {
            position: relative;
            margin: 20px 0;
        }

        .file-input {
            display: none;
        }

        .file-label {
            display: inline-block;
            padding: 15px 40px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 50px;
            cursor: pointer;
            font-size: 1.1em;
            font-weight: 600;
            transition: all 0.3s ease;
        }

        .file-label:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 20px rgba(102, 126, 234, 0.4);
        }

        .file-name {
            margin-top: 15px;
            font-size: 0.95em;
            color: #666;
            font-weight: 500;
        }

        .model-selection {
            margin: 20px 0;
            text-align: center;
        }

        .model-selection label {
            display: inline-block;
            margin: 0 15px;
            font-size: 1em;
            cursor: pointer;
        }

        .model-selection input[type="radio"] {
            margin-right: 5px;
        }

        .submit-btn {
            width: 100%;
            padding: 18px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 50px;
            font-size: 1.2em;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            margin-top: 20px;
        }

        .submit-btn:hover:not(:disabled) {
            transform: translateY(-2px);
            box-shadow: 0 10px 30px rgba(102, 126, 234, 0.4);
        }

        .submit-btn:disabled {
            opacity: 0.6;
            cursor: not-allowed;
        }

        .results {
            display: none;
            margin-top: 30px;
            animation: fadeIn 0.5s ease;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .results-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px 10px 0 0;
            font-size: 1.3em;
            font-weight: 600;
        }

        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            padding: 30px;
            background: #f8f9fa;
        }

        .stat-card {
            background: white;
            padding: 25px;
            border-radius: 10px;
            text-align: center;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            transition: all 0.3s ease;
        }

        .stat-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 8px 15px rgba(0, 0, 0, 0.2);
        }

        .stat-value {
            font-size: 2.5em;
            font-weight: bold;
            margin: 10px 0;
        }

        .stat-value.fraud {
            color: #e74c3c;
        }

        .stat-value.legitimate {
            color: #27ae60;
        }

        .stat-value.total {
            color: #3498db;
        }

        .stat-label {
            color: #666;
            font-size: 0.95em;
            text-transform: uppercase;
            letter-spacing: 1px;
        }

        .high-risk-section {
            padding: 30px;
            background: white;
        }

        .high-risk-title {
            font-size: 1.3em;
            color: #333;
            margin-bottom: 20px;
            font-weight: 600;
        }

        .risk-transaction {
            background: #fff5f5;
            border-left: 4px solid #e74c3c;
            padding: 15px;
            margin-bottom: 15px;
            border-radius: 5px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .risk-info {
            flex: 1;
        }

        .risk-probability {
            background: #e74c3c;
            color: white;
            padding: 8px 15px;
            border-radius: 20px;
            font-weight: 600;
        }

        .loading {
            text-align: center;
            padding: 20px;
            display: none;
        }

        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #667eea;
            border-radius: 50%;
            width: 50px;
            height: 50px;
            animation: spin 1s linear infinite;
            margin: 0 auto;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        .error {
            background: #fee;
            border-left: 4px solid #e74c3c;
            color: #c33;
            padding: 15px;
            border-radius: 5px;
            margin-top: 20px;
            display: none;
        }

        .info-box {
            background: #e8f4fd;
            border-left: 4px solid #3498db;
            padding: 20px;
            border-radius: 5px;
            margin-bottom: 30px;
        }

        .info-box h3 {
            color: #2980b9;
            margin-bottom: 10px;
        }

        .info-box ul {
            margin-left: 20px;
            color: #555;
        }

        .info-box li {
            margin: 5px 0;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🛡️ Fraud Detection System</h1>
            <p>AI-Powered Credit Card Transaction Analysis</p>
        </div>

        <div class="content">
            <div class="info-box">
                <h3>📋 Instructions</h3>
                <ul>
                    <li>Upload your credit card transaction data in CSV or PDF format</li>
                    <li>File must contain columns: Time, V1-V28, Amount</li>
                    <li>Select your preferred ML model for analysis</li>
                    <li>Get instant fraud detection results</li>
                </ul>
            </div>

            <form id="uploadForm" enctype="multipart/form-data">
                <div class="upload-section" id="dropZone">
                    <h2>📁 Upload Transaction Data</h2>
                    <p style="margin: 10px 0; color: #666;">Drag and drop your file here or click to browse</p>
                    
                    <div class="file-input-wrapper">
                        <input type="file" id="fileInput" name="file" class="file-input" accept=".csv,.pdf" required>
                        <label for="fileInput" class="file-label">Choose File</label>
                    </div>
                    
                    <div id="fileName" class="file-name"></div>
                </div>

                <div class="model-selection">
                    <h3 style="margin-bottom: 15px;">Select Model:</h3>
                    <label>
                        <input type="radio" name="model" value="tuned" checked>
                        Tuned Model (Optimized)
                    </label>
                    <label>
                        <input type="radio" name="model" value="rf">
                        Random Forest Model
                    </label>
                </div>

                <button type="submit" class="submit-btn" id="submitBtn">
                    🔍 Analyze Transactions
                </button>
            </form>

            <div class="loading" id="loading">
                <div class="spinner"></div>
                <p style="margin-top: 15px; color: #666;">Analyzing transactions...</p>
            </div>

            <div class="error" id="error"></div>

            <div class="results" id="results">
                <div class="results-header">
                    📊 Analysis Results
                </div>
                
                <div class="stats-grid">
                    <div class="stat-card">
                        <div class="stat-label">Total Transactions</div>
                        <div class="stat-value total" id="totalTransactions">0</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-label">Fraudulent</div>
                        <div class="stat-value fraud" id="fraudCount">0</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-label">Legitimate</div>
                        <div class="stat-value legitimate" id="legitimateCount">0</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-label">Fraud Rate</div>
                        <div class="stat-value fraud" id="fraudPercentage">0%</div>
                    </div>
                </div>

                <div class="high-risk-section" id="highRiskSection">
                    <div class="high-risk-title">⚠️ High-Risk Transactions</div>
                    <div id="highRiskList"></div>
                </div>
            </div>
        </div>
    </div>

    <script>
        const form = document.getElementById('uploadForm');
        const fileInput = document.getElementById('fileInput');
        const fileName = document.getElementById('fileName');
        const loading = document.getElementById('loading');
        const results = document.getElementById('results');
        const error = document.getElementById('error');
        const dropZone = document.getElementById('dropZone');

        // File input change
        fileInput.addEventListener('change', function(e) {
            if (e.target.files.length > 0) {
                fileName.textContent = `Selected: ${e.target.files[0].name}`;
            }
        });

        // Drag and drop
        dropZone.addEventListener('dragover', function(e) {
            e.preventDefault();
            dropZone.classList.add('dragover');
        });

        dropZone.addEventListener('dragleave', function(e) {
            e.preventDefault();
            dropZone.classList.remove('dragover');
        });

        dropZone.addEventListener('drop', function(e) {
            e.preventDefault();
            dropZone.classList.remove('dragover');
            
            if (e.dataTransfer.files.length > 0) {
                fileInput.files = e.dataTransfer.files;
                fileName.textContent = `Selected: ${e.dataTransfer.files[0].name}`;
            }
        });

        // Form submission
        form.addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const formData = new FormData(form);
            
            // Hide previous results/errors
            results.style.display = 'none';
            error.style.display = 'none';
            loading.style.display = 'block';
            
            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    body: formData
                });
                
                const data = await response.json();
                
                loading.style.display = 'none';
                
                if (!response.ok) {
                    throw new Error(data.error || 'An error occurred');
                }
                
                // Display results
                document.getElementById('totalTransactions').textContent = data.total_transactions;
                document.getElementById('fraudCount').textContent = data.fraud_count;
                document.getElementById('legitimateCount').textContent = data.legitimate_count;
                document.getElementById('fraudPercentage').textContent = data.fraud_percentage + '%';
                
                // Display high-risk transactions
                const highRiskList = document.getElementById('highRiskList');
                if (data.high_risk_transactions.length > 0) {
                    highRiskList.innerHTML = data.high_risk_transactions.map(tx => `
                        <div class="risk-transaction">
                            <div class="risk-info">
                                <strong>Transaction #${tx.index + 1}</strong><br>
                                Amount: $${typeof tx.amount === 'number' ? tx.amount.toFixed(2) : tx.amount}
                            </div>
                            <div class="risk-probability">
                                ${(tx.probability * 100).toFixed(1)}% Risk
                            </div>
                        </div>
                    `).join('');
                } else {
                    highRiskList.innerHTML = '<p style="color: #27ae60; font-weight: 600;">✓ No high-risk transactions detected!</p>';
                }
                
                results.style.display = 'block';
                results.scrollIntoView({ behavior: 'smooth' });
                
            } catch (err) {
                loading.style.display = 'none';
                error.textContent = '❌ ' + err.message;
                error.style.display = 'block';
            }
        });
    </script>
</body>
</html>'''
    
    with open('templates/index.html', 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print("\n" + "="*60)
    print("🚀 Fraud Detection Web App Starting...")
    print("="*60)
    print("\n📂 Models Location: C:\\fraud detection trained models all files")
    print("\n✓ Application ready!")
    print("\n🌐 Open your browser and go to: http://127.0.0.1:5000")
    print("\n" + "="*60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)