from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the model and scaler you saved earlier
model = joblib.load('isolation_forest_v1.joblib')
scaler = joblib.load('scaler_v1.joblib')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        # Extract features in the correct order
        features = np.array([[
            data['current'], 
            data['temperature'], 
            data['vibration']
        ]])
        
        # 1. Scale the data
        features_scaled = scaler.transform(features)
        
        # 2. Predict (-1 = anomaly, 1 = normal)
        prediction = model.predict(features_scaled)
        
        # Convert to boolean for your ESP32/Dashboard
        is_anomaly = True if prediction[0] == -1 else False
        
        return jsonify({
            "is_anomaly": is_anomaly,
            "status": "success"
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)