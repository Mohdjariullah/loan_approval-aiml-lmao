"""
Loan Approval using KNN - Flask Web Application
This Flask app provides a web interface for loan approval predictions.
"""

from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load model, scaler, encoders, and feature names
MODEL_PATH = 'loan_approval/models/knn_model.pkl'
SCALER_PATH = 'loan_approval/models/scaler.pkl'
ENCODERS_PATH = 'loan_approval/models/encoders.pkl'
FEATURE_NAMES_PATH = 'loan_approval/models/feature_names.pkl'

def load_model():
    """Load the trained model, scaler, encoders, and feature names."""
    if not os.path.exists(MODEL_PATH):
        return None, None, None, None
    
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    encoders = joblib.load(ENCODERS_PATH)
    feature_names = joblib.load(FEATURE_NAMES_PATH)
    
    return model, scaler, encoders, feature_names

model, scaler, encoders, feature_names = load_model()

@app.route('/')
def index():
    """Render the home page with input form."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests."""
    if model is None or scaler is None or encoders is None or feature_names is None:
        return jsonify({'error': 'Model not found. Please train the model first by running train_model.py'}), 500
    
    try:
        # Get form data
        data = request.form.to_dict()
        
        # Create feature array in correct order
        features = []
        for feature_name in feature_names:
            value = data.get(feature_name, '')
            
            # Encode categorical features
            if feature_name in encoders:
                try:
                    encoded_value = encoders[feature_name].transform([str(value)])[0]
                    features.append(encoded_value)
                except ValueError:
                    # If value not in encoder, use most common value
                    encoded_value = encoders[feature_name].transform([encoders[feature_name].classes_[0]])[0]
                    features.append(encoded_value)
            else:
                # Numerical feature
                features.append(float(value))
        
        # Convert to numpy array and reshape
        features_array = np.array(features).reshape(1, -1)
        
        # Scale features
        features_scaled = scaler.transform(features_array)
        
        # Make prediction
        prediction = model.predict(features_scaled)[0]
        prediction_proba = model.predict_proba(features_scaled)[0]
        
        # Get confidence (probability of predicted class)
        confidence = prediction_proba[prediction] * 100
        
        # Prepare result
        result = {
            'prediction': int(prediction),
            'prediction_text': 'Loan Approved' if prediction == 1 else 'Loan Not Approved',
            'confidence': round(confidence, 2),
            'probabilities': {
                'not_approved': round(prediction_proba[0] * 100, 2),
                'approved': round(prediction_proba[1] * 100, 2)
            }
        }
        
        return render_template('result.html', result=result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for predictions (returns JSON)."""
    if model is None or scaler is None or encoders is None or feature_names is None:
        return jsonify({'error': 'Model not found. Please train the model first by running train_model.py'}), 500
    
    try:
        # Get JSON data
        data = request.get_json()
        
        # Create feature array in correct order
        features = []
        for feature_name in feature_names:
            value = data.get(feature_name, '')
            
            # Encode categorical features
            if feature_name in encoders:
                try:
                    encoded_value = encoders[feature_name].transform([str(value)])[0]
                    features.append(encoded_value)
                except ValueError:
                    # If value not in encoder, use most common value
                    encoded_value = encoders[feature_name].transform([encoders[feature_name].classes_[0]])[0]
                    features.append(encoded_value)
            else:
                # Numerical feature
                features.append(float(value))
        
        # Convert to numpy array and reshape
        features_array = np.array(features).reshape(1, -1)
        
        # Scale features
        features_scaled = scaler.transform(features_array)
        
        # Make prediction
        prediction = model.predict(features_scaled)[0]
        prediction_proba = model.predict_proba(features_scaled)[0]
        
        # Get confidence (probability of predicted class)
        confidence = prediction_proba[prediction] * 100
        
        # Prepare result
        result = {
            'prediction': int(prediction),
            'prediction_text': 'Loan Approved' if prediction == 1 else 'Loan Not Approved',
            'confidence': round(confidence, 2),
            'probabilities': {
                'not_approved': round(prediction_proba[0] * 100, 2),
                'approved': round(prediction_proba[1] * 100, 2)
            }
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    if model is None:
        print("WARNING: Model not found. Please run train_model.py first to train the model.")
    app.run(debug=True, host='0.0.0.0', port=5001)

