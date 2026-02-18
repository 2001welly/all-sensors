import numpy as np
import joblib
import os
from flask import Flask, request, jsonify

# Initialize Flask App
app = Flask(__name__)

# 1. Load the saved model and scaler
# Ensure these files are in the same directory as app.py
try:
    model = joblib.load("isolation_forest_model.joblib")
    scaler = joblib.load("scaler.joblib")
    print("Model and Scaler loaded successfully.")
except Exception as e:
    print(f"Error loading model files: {e}")

@app.route('/')
def home():
    return {
        "message": "Sensor Anomaly Detection API is live!",
        "usage": "Send a POST request to /predict with JSON data."
    }

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 2. Get JSON data from request
        # Expected format: {"features": [[current, temp, other_val]]}
        data = request.get_json()
        
        if not data or 'features' not in data:
            return jsonify({"error": "No features provided in request"}), 400
            
        features = np.array(data['features'])
        
        # 3. Scale the input using the loaded scaler
        features_scaled = scaler.transform(features)
        
        # 4. Get Raw Predictions and Scores
        # model.predict returns 1 (Normal) or -1 (Anomaly)
        raw_predictions = model.predict(features_scaled)
        
        # model.decision_function returns negative for anomaly, positive for normal
        anomaly_scores = model.decision_function(features_scaled)
        
        # 5. Process Results
        final_results = []
        for pred, score in zip(raw_predictions, anomaly_scores):
            # THE STANDARD SCIKIT-LEARN WAY:
            # -1 is Anomaly, 1 is Normal
            if pred == -1:
                label = "Anomaly"
                is_anomaly = True
            else:
                label = "Normal"
                is_anomaly = False
                
            final_results.append({
                "prediction_label": label,
                "is_anomaly": is_anomaly,
                "score": round(float(score), 4),
                "description": "Negative score indicates anomaly"
            })
            
        return jsonify({
            "status": "success",
            "results": final_results
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

if __name__ == '__main__':
    # Render provides a PORT environment variable
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)