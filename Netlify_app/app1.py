<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Fraud Detection System</title>
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
            overflow-x: hidden;
        }

        /* Loading Screen Styles */
        #loading-screen {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            z-index: 9999;
            transition: opacity 0.5s, visibility 0.5s;
        }

        #loading-screen.hidden {
            opacity: 0;
            visibility: hidden;
        }

        #svg-global {
            zoom: 1.2;
            overflow: visible;
        }

        @keyframes fade-particles {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }

        @keyframes floatUp {
            0% { transform: translateY(0); opacity: 0; }
            10% { opacity: 1; }
            100% { transform: translateY(-40px); opacity: 0; }
        }

        #particles {
            animation: fade-particles 5s infinite alternate;
        }

        .particle {
            animation: floatUp linear infinite;
        }

        .p1 { animation-duration: 2.2s; animation-delay: 0s; }
        .p2 { animation-duration: 2.5s; animation-delay: 0.3s; }
        .p3 { animation-duration: 2s; animation-delay: 0.6s; }
        .p4 { animation-duration: 2.8s; animation-delay: 0.2s; }
        .p5 { animation-duration: 2.3s; animation-delay: 0.4s; }
        .p6 { animation-duration: 3s; animation-delay: 0.1s; }
        .p7 { animation-duration: 2.1s; animation-delay: 0.5s; }
        .p8 { animation-duration: 2.6s; animation-delay: 0.2s; }
        .p9 { animation-duration: 2.4s; animation-delay: 0.3s; }

        @keyframes bounce-lines {
            0%, 100% { transform: translateY(0); }
            50% { transform: translateY(-3px); }
        }

        #line-v1, #line-v2, #node-server, #panel-rigth, #reflectores, #particles {
            animation: bounce-lines 3s ease-in-out infinite alternate;
        }
        #line-v2 { animation-delay: 0.2s; }
        #node-server, #panel-rigth, #reflectores, #particles { animation-delay: 0.4s; }

        .loading-text {
            color: white;
            font-size: 24px;
            margin-top: 30px;
            font-weight: 600;
            letter-spacing: 2px;
        }

        /* Main App Styles */
        #main-app {
            opacity: 0;
            transition: opacity 0.5s;
            padding: 20px;
        }

        #main-app.visible {
            opacity: 1;
        }

        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            padding: 40px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        }

        h1 {
            color: #667eea;
            text-align: center;
            font-size: 2.5em;
            margin-bottom: 10px;
        }

        .subtitle {
            text-align: center;
            color: #764ba2;
            font-size: 1.2em;
            margin-bottom: 30px;
            font-weight: 600;
        }

        .info-box {
            background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%);
            border-left: 4px solid #667eea;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 30px;
        }

        .info-box h3 {
            color: #667eea;
            margin-bottom: 10px;
        }

        .info-box ul {
            margin-left: 20px;
            color: #555;
            line-height: 1.8;
        }

        .upload-area {
            border: 3px dashed #667eea;
            border-radius: 15px;
            padding: 40px;
            text-align: center;
            margin: 30px 0;
            background: #f8f9ff;
            cursor: pointer;
            transition: all 0.3s;
        }

        .upload-area:hover {
            background: #f0f2ff;
            border-color: #764ba2;
        }

        .upload-area.dragover {
            background: #e8ebff;
            border-color: #764ba2;
            transform: scale(1.02);
        }

        input[type="file"] {
            display: none;
        }

        .upload-icon {
            font-size: 48px;
            color: #667eea;
            margin-bottom: 15px;
        }

        .btn {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 15px 40px;
            font-size: 18px;
            border-radius: 50px;
            cursor: pointer;
            transition: all 0.3s;
            font-weight: 600;
            letter-spacing: 1px;
            width: 100%;
            margin-top: 20px;
        }

        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 30px rgba(102, 126, 234, 0.4);
        }

        .btn:disabled {
            opacity: 0.6;
            cursor: not-allowed;
        }

        .results {
            margin-top: 40px;
            display: none;
        }

        .results.visible {
            display: block;
        }

        .metrics {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 30px 0;
        }

        .metric-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 25px;
            border-radius: 15px;
            text-align: center;
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        }

        .metric-value {
            font-size: 2.5em;
            font-weight: bold;
            margin-bottom: 10px;
        }

        .metric-label {
            font-size: 0.9em;
            opacity: 0.9;
        }

        .alert {
            padding: 15px 20px;
            border-radius: 10px;
            margin: 20px 0;
        }

        .alert-success {
            background: #d4edda;
            color: #155724;
            border-left: 4px solid #28a745;
        }

        .alert-warning {
            background: #fff3cd;
            color: #856404;
            border-left: 4px solid #ffc107;
        }

        .alert-info {
            background: #d1ecf1;
            color: #0c5460;
            border-left: 4px solid #17a2b8;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .fade-in {
            animation: fadeIn 0.5s ease-out;
        }
    </style>
</head>
<body>
    <!-- Loading Screen -->
    <div id="loading-screen">
        <svg id="svg-global" width="200" height="200" viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
            <defs>
                <linearGradient id="grad1" x1="0%" y1="0%" x2="100%" y2="100%">
                    <stop offset="0%" style="stop-color:#667eea;stop-opacity:1" />
                    <stop offset="100%" style="stop-color:#764ba2;stop-opacity:1" />
                </linearGradient>
            </defs>
            
            <g id="particles">
                <circle class="particle p1" cx="30" cy="70" r="2" fill="white" opacity="0.8"/>
                <circle class="particle p2" cx="40" cy="75" r="1.5" fill="white" opacity="0.6"/>
                <circle class="particle p3" cx="50" cy="70" r="2" fill="white" opacity="0.8"/>
                <circle class="particle p4" cx="60" cy="75" r="1.5" fill="white" opacity="0.6"/>
                <circle class="particle p5" cx="70" cy="70" r="2" fill="white" opacity="0.8"/>
                <circle class="particle p6" cx="35" cy="72" r="1" fill="white" opacity="0.5"/>
                <circle class="particle p7" cx="55" cy="73" r="1" fill="white" opacity="0.5"/>
                <circle class="particle p8" cx="65" cy="72" r="1" fill="white" opacity="0.5"/>
                <circle class="particle p9" cx="45" cy="73" r="1" fill="white" opacity="0.5"/>
            </g>
            
            <rect id="node-server" x="35" y="30" width="30" height="35" rx="3" fill="white" stroke="url(#grad1)" stroke-width="2"/>
            <line id="line-v1" x1="50" y1="65" x2="50" y2="75" stroke="white" stroke-width="2"/>
            <circle cx="50" cy="20" r="8" fill="white" stroke="url(#grad1)" stroke-width="2"/>
            <text x="50" y="24" text-anchor="middle" fill="url(#grad1)" font-size="10" font-weight="bold">AI</text>
        </svg>
        <div class="loading-text">Loading Fraud Detection System...</div>
    </div>

    <!-- Main App -->
    <div id="main-app">
        <div class="container">
            <h1>🛡️ Fraud Detection System</h1>
            <div class="subtitle">CREDIT CARD TRANSACTION ANALYSIS</div>

            <div class="info-box">
                <h3>📋 How to Use:</h3>
                <ul>
                    <li>Upload your credit card transaction data (CSV or PDF)</li>
                    <li>File must have columns: Time, V1-V28, Amount</li>
                    <li>Optional: Include 'Class' column (0=Legitimate, 1=Fraud) for performance metrics</li>
                    <li>Click "Analyze Transactions" to detect fraud with 3 AI models</li>
                    <li>Get instant results with comprehensive fraud risk analysis!</li>
                </ul>
            </div>

            <div class="upload-area" id="uploadArea">
                <div class="upload-icon">📁</div>
                <h3>Drop your file here or click to browse</h3>
                <p style="color: #666; margin-top: 10px;">Supports CSV and PDF files</p>
                <input type="file" id="fileInput" accept=".csv,.pdf">
                <div id="fileName" style="margin-top: 15px; color: #667eea; font-weight: 600;"></div>
            </div>

            <button class="btn" id="analyzeBtn" disabled>Analyze Transactions</button>

            <div class="results" id="results">
                <h2 style="color: #667eea; margin-top: 40px;">📊 Analysis Results</h2>
                
                <div class="metrics">
                    <div class="metric-card">
                        <div class="metric-value" id="totalTransactions">0</div>
                        <div class="metric-label">Total Transactions</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value" id="fraudDetected">0</div>
                        <div class="metric-label">Fraud Detected</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value" id="fraudPercent">0%</div>
                        <div class="metric-label">Fraud Rate</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value" id="riskScore">0.00</div>
                        <div class="metric-label">Risk Score</div>
                    </div>
                </div>

                <div id="alertsContainer"></div>

                <div style="text-align: center; margin-top: 40px; padding: 30px; background: #f8f9ff; border-radius: 15px;">
                    <h3 style="color: #667eea; margin-bottom: 15px;">🔗 Full ML Analysis Available</h3>
                    <p style="color: #666; line-height: 1.6;">
                        This is a demo interface. For complete fraud detection with trained ML models (Tuned Model, Random Forest, XGBoost), 
                        deploy the full Streamlit application to Streamlit Cloud, Heroku, or Railway.
                    </p>
                    <p style="color: #666; margin-top: 10px;">
                        <strong>Next Steps:</strong> Host the backend API and connect it to this frontend for real-time fraud detection.
                    </p>
                </div>
            </div>
        </div>
    </div>

    <script>
        // Hide loading screen after page loads
        window.addEventListener('load', () => {
            setTimeout(() => {
                document.getElementById('loading-screen').classList.add('hidden');
                document.getElementById('main-app').classList.add('visible');
            }, 2000);
        });

        const uploadArea = document.getElementById('uploadArea');
        const fileInput = document.getElementById('fileInput');
        const fileName = document.getElementById('fileName');
        const analyzeBtn = document.getElementById('analyzeBtn');
        const results = document.getElementById('results');

        let selectedFile = null;

        uploadArea.addEventListener('click', () => fileInput.click());

        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.classList.add('dragover');
        });

        uploadArea.addEventListener('dragleave', () => {
            uploadArea.classList.remove('dragover');
        });

        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                handleFile(files[0]);
            }
        });

        fileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                handleFile(e.target.files[0]);
            }
        });

        function handleFile(file) {
            const validExtensions = ['csv', 'pdf'];
            const ext = file.name.split('.').pop().toLowerCase();
            
            if (validExtensions.includes(ext)) {
                selectedFile = file;
                fileName.textContent = `✓ Selected: ${file.name}`;
                analyzeBtn.disabled = false;
            } else {
                fileName.textContent = '❌ Invalid file type. Please upload CSV or PDF.';
                fileName.style.color = '#dc3545';
                analyzeBtn.disabled = true;
            }
        }

        analyzeBtn.addEventListener('click', () => {
            if (!selectedFile) return;

            // Simulate analysis (in production, this would call your backend API)
            analyzeBtn.textContent = 'Analyzing...';
            analyzeBtn.disabled = true;

            setTimeout(() => {
                // Demo data
                const totalTrans = Math.floor(Math.random() * 500) + 100;
                const fraudCount = Math.floor(Math.random() * 50) + 5;
                const fraudPct = ((fraudCount / totalTrans) * 100).toFixed(2);
                const riskScore = (Math.random() * 0.5 + 0.3).toFixed(2);

                document.getElementById('totalTransactions').textContent = totalTrans;
                document.getElementById('fraudDetected').textContent = fraudCount;
                document.getElementById('fraudPercent').textContent = fraudPct + '%';
                document.getElementById('riskScore').textContent = riskScore;

                const alertsContainer = document.getElementById('alertsContainer');
                alertsContainer.innerHTML = '';

                if (fraudCount > 30) {
                    alertsContainer.innerHTML += `
                        <div class="alert alert-warning fade-in">
                            <strong>⚠️ High Risk Alert!</strong> Detected ${fraudCount} potentially fraudulent transactions requiring immediate review.
                        </div>
                    `;
                } else if (fraudCount > 10) {
                    alertsContainer.innerHTML += `
                        <div class="alert alert-info fade-in">
                            <strong>ℹ️ Medium Risk:</strong> Found ${fraudCount} suspicious transactions. Recommend further investigation.
                        </div>
                    `;
                } else {
                    alertsContainer.innerHTML += `
                        <div class="alert alert-success fade-in">
                            <strong>✓ Low Risk:</strong> Only ${fraudCount} potentially fraudulent transactions detected.
                        </div>
                    `;
                }

                results.classList.add('visible');
                results.scrollIntoView({ behavior: 'smooth' });

                analyzeBtn.textContent = 'Analyze Transactions';
                analyzeBtn.disabled = false;
            }, 2000);
        });
    </script>
</body>
</html>
