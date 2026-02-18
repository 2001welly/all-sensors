import numpy as np
import joblib
from flask import Flask, request, jsonify

# Initialize Flask App
app = Flask(__name__)

# Load the saved model and scaler
model = joblib.load("isolation_forest_model.joblib")
scaler = joblib.load("scaler.joblib")

@app.route('/')
def home():
    return "Isolation Forest Anomaly Detection API is Running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from request
        data = request.get_json()
        
        # Ensure data is in a 2D array format for the scaler
        # Example input: {"features": [[23.5, 101.2, 0.5]]}
        features = np.array(data['features'])
        
        # 1. Scale the input
        features_scaled = scaler.transform(features)
        
        # 2. Get Prediction (-1 for anomaly, 1 for normal)
        prediction = model.predict(features_scaled)
        
        # 3. Get Anomaly Score (Lower values = more anomalous)
        score = model.decision_function(features_scaled)
        
        # Convert prediction to 1 (anomaly) and 0 (normal)
        results = []
        for p, s in zip(prediction, score):
            results.append({
                "prediction": "anomaly" if p == -1 else "normal",
                "is_anomaly": int(p == -1),
                "anomaly_score": float(s)
            })
            
        return jsonify({"status": "success", "results": results})

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

if __name__ == '__main__':
    # Use port 5000 by default, but Render will provide a PORT env variable
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)