from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load artifacts
model = joblib.load("isolation_forest_model.joblib")
scaler = joblib.load("scaler.joblib")
feature_columns = joblib.load("feature_columns.joblib")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    
    # Ensure correct order
    input_data = [data[col] for col in feature_columns]
    
    # Convert to numpy array
    input_array = np.array(input_data).reshape(1, -1)
    
    # Scale
    scaled_data = scaler.transform(input_array)
    
    # Predict
    prediction = model.predict(scaled_data)
    
    result = "Anomaly" if prediction[0] == -1 else "Normal"
    
    return jsonify({
        "prediction": result
    })

if __name__ == "__main__":
    app.run()
