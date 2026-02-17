from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the saved model and scaler
try:
    model = joblib.load('isolation_forest_v1.joblib')
    scaler = joblib.load('scaler_v1.joblib')
    print("AI Model & Scaler loaded successfully.")
except Exception as e:
    print(f"Error loading model files: {e}")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        # Features must be in the exact order: current, temperature, vibration
        current = data.get('current', 0)
        temp = data.get('temperature', 0)
        vibe = data.get('vibration', 0)
        
        features = np.array([[current, temp, vibe]])
        
        # 1. Scale the input data
        features_scaled = scaler.transform(features)
        
        # 2. Calculate the raw Anomaly Score
        # Isolation Forest: score < 0 is an Anomaly
        score = model.decision_function(features_scaled)[0]
        is_anomaly = bool(score < 0)
        
        # Log results to Render console for monitoring
        print(f"In: {current}A, {temp}C, {vibe}g | Score: {score:.4f} | Anomaly: {is_anomaly}")

        return jsonify({
            "is_anomaly": is_anomaly,
            "score": float(score),
            "status": "success"
        })
    
    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({"is_anomaly": False, "error": str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)