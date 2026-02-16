from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the model trained on three features (current, temperature, vibration)
# Ensure this file is in the same folder as app.py
model = joblib.load('isolation_forest_model.joblib')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from the ESP32
        data = request.get_json()

        # Extract three required values
        current_val = data.get('current')
        temp_val = data.get('temperature')
        vibe_val = data.get('vibration')

        # Validate that all three are present
        if None in (current_val, temp_val, vibe_val):
            return jsonify({
                'error': 'Missing one or more values. Expected keys: current, temperature, vibration'
            }), 400

        # Convert to numpy array (shape 1x3) for prediction
        input_data = np.array([[float(current_val), float(temp_val), float(vibe_val)]])

        # Predict: 1 = normal, -1 = anomaly
        prediction = model.predict(input_data)[0]
        is_anomaly = bool(prediction == -1)

        # Return result
        return jsonify({
            'current': current_val,
            'temperature': temp_val,
            'vibration': vibe_val,
            'status': 'Anomaly' if is_anomaly else 'Normal',
            'is_anomaly': is_anomaly
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # For local testing
    app.run(host='0.0.0.0', port=5000)