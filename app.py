from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the model you created in Step 1
# Ensure this file is in the same folder as app.py
model = joblib.load('isolation_forest_model.joblib')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the JSON data from your sensor/hardware
        data = request.get_json()
        current_val = data.get('current')
        
        if current_val is None:
            return jsonify({'error': 'No current value provided'}), 400

        # Convert to numpy array for the model
        input_data = np.array([[float(current_val)]])
        
        # Predict: 1 = Normal, -1 = Anomaly
        prediction = model.predict(input_data)[0]
        
        return jsonify({
            'current': current_val,
            'status': 'Anomaly' if prediction == -1 else 'Normal',
            'is_anomaly': bool(prediction == -1)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Render will use a production server (gunicorn), 
    # but this allows for local testing.
    app.run(host='0.0.0.0', port=5000)