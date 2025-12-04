"""
Loan Approval using KNN - Streamlit Web Application
This Streamlit app provides a web interface for loan approval predictions.
Deployment-ready for Streamlit Cloud.
"""

import streamlit as st
import joblib
import numpy as np
import os
import warnings
from pathlib import Path

# Suppress scikit-learn version warnings (models work fine across versions)
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

# Page configuration
st.set_page_config(
    page_title="Loan Approval Prediction System",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #f093fb 0%, #f5576c 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #f093fb 0%, #f5576c 100%);
        color: white;
        font-weight: bold;
        padding: 0.75rem;
        border-radius: 5px;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        margin-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Model paths
@st.cache_resource
def load_model():
    """Load the trained model, scaler, encoders, and feature names with caching."""
    base_path = Path(__file__).parent
    
    # Try different path configurations for deployment
    possible_model_paths = [
        base_path / 'loan_approval' / 'models' / 'knn_model.pkl',  # From root
        base_path / 'models' / 'knn_model.pkl',  # Direct models folder
        Path('loan_approval/models/knn_model.pkl'),  # Absolute from root
        Path('loan_approval/loan_approval/models/knn_model.pkl'),  # Nested structure
    ]
    
    possible_scaler_paths = [
        base_path / 'loan_approval' / 'models' / 'scaler.pkl',
        base_path / 'models' / 'scaler.pkl',
        Path('loan_approval/models/scaler.pkl'),
        Path('loan_approval/loan_approval/models/scaler.pkl'),
    ]
    
    possible_encoder_paths = [
        base_path / 'loan_approval' / 'models' / 'encoders.pkl',
        base_path / 'models' / 'encoders.pkl',
        Path('loan_approval/models/encoders.pkl'),
        Path('loan_approval/loan_approval/models/encoders.pkl'),
    ]
    
    possible_feature_paths = [
        base_path / 'loan_approval' / 'models' / 'feature_names.pkl',
        base_path / 'models' / 'feature_names.pkl',
        Path('loan_approval/models/feature_names.pkl'),
        Path('loan_approval/loan_approval/models/feature_names.pkl'),
    ]
    
    model_path = None
    scaler_path = None
    encoder_path = None
    feature_path = None
    
    for path in possible_model_paths:
        if path.exists():
            model_path = path
            break
    
    for path in possible_scaler_paths:
        if path.exists():
            scaler_path = path
            break
    
    for path in possible_encoder_paths:
        if path.exists():
            encoder_path = path
            break
    
    for path in possible_feature_paths:
        if path.exists():
            feature_path = path
            break
    
    if model_path and scaler_path and encoder_path and feature_path:
        try:
            model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)
            encoders = joblib.load(encoder_path)
            feature_names = joblib.load(feature_path)
            return model, scaler, encoders, feature_names, None
        except Exception as e:
            return None, None, None, None, str(e)
    else:
        return None, None, None, None, "Model files not found. Please ensure models are trained."

# Load model
model, scaler, encoders, feature_names, error = load_model()

# Main header
st.markdown("""
<div class="main-header">
    <h1>💰 Loan Approval Prediction System</h1>
    <p>Using K-Nearest Neighbors (KNN) Machine Learning Algorithm</p>
</div>
""", unsafe_allow_html=True)

# Sidebar with information
with st.sidebar:
    st.header("📋 Information")
    st.info("""
    This system uses K-Nearest Neighbors (KNN) machine learning algorithm 
    to predict loan approval based on applicant information, financial details, 
    and credit history.
    
    **Note:** This is a demonstration system. Actual loan approval decisions 
    are made by financial institutions based on comprehensive evaluation.
    """)
    
    if error:
        st.error(f"⚠️ {error}")
    elif model is not None:
        st.success("✅ Model loaded successfully!")

# Main content
if model is None or scaler is None or encoders is None or feature_names is None:
    st.error("""
    ⚠️ **Model Not Found**
    
    Please train the model first by running:
    ```bash
    python train_model.py
    ```
    
    Make sure the model files are located in `loan_approval/models/` directory:
    - `knn_model.pkl`
    - `scaler.pkl`
    - `encoders.pkl`
    - `feature_names.pkl`
    """)
    st.stop()

# Create form for input
st.header("📝 Applicant Information")
st.markdown("Please enter the applicant's information and financial details:")

# Create columns for better layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("Personal Information")
    gender = st.selectbox("Gender", options=["Male", "Female"])
    married = st.selectbox("Married", options=["Yes", "No"])
    dependents = st.selectbox("Dependents", options=["0", "1", "2", "3+"])
    education = st.selectbox("Education", options=["Graduate", "Not Graduate"])
    self_employed = st.selectbox("Self Employed", options=["Yes", "No"])
    property_area = st.selectbox("Property Area", options=["Urban", "Semiurban", "Rural"])

with col2:
    st.subheader("Financial Information")
    applicant_income = st.number_input("Applicant Income (₹)", min_value=0, value=5000, step=100)
    coapplicant_income = st.number_input("Coapplicant Income (₹)", min_value=0, value=0, step=100)
    loan_amount = st.number_input("Loan Amount (₹)", min_value=0, value=50000, step=1000)
    loan_amount_term = st.selectbox(
        "Loan Amount Term (months)", 
        options=[12, 36, 60, 84, 120, 180, 240, 300, 360]
    )
    credit_history = st.selectbox(
        "Credit History", 
        options=["Good (1)", "Not Good (0)"],
        format_func=lambda x: x
    )

# Convert credit history
credit_history_value = 1 if credit_history == "Good (1)" else 0

# Prediction button
st.markdown("---")
predict_button = st.button("🔍 Predict Loan Approval", type="primary", use_container_width=True)

if predict_button:
    try:
        # Prepare features dictionary
        feature_dict = {
            'Gender': gender,
            'Married': married,
            'Dependents': dependents,
            'Education': education,
            'Self_Employed': self_employed,
            'ApplicantIncome': applicant_income,
            'CoapplicantIncome': coapplicant_income,
            'LoanAmount': loan_amount,
            'Loan_Amount_Term': loan_amount_term,
            'Credit_History': credit_history_value,
            'Property_Area': property_area
        }
        
        # Debug: Check for missing features (only show if there are issues)
        missing_features = [f for f in feature_names if f not in feature_dict]
        if missing_features:
            st.warning(f"⚠️ Model expects features not in form: {missing_features}")
            st.info(f"📋 Model feature names: {list(feature_names)}")
            st.info(f"📋 Form feature names: {list(feature_dict.keys())}")
        
        # Create feature array in correct order
        features = []
        for feature_name in feature_names:
            value = feature_dict.get(feature_name, None)
            
            # Handle missing features
            if value is None or value == '':
                # Provide default values based on feature type
                if feature_name in encoders:
                    # For categorical features, use first class as default
                    encoded_value = encoders[feature_name].transform([encoders[feature_name].classes_[0]])[0]
                    features.append(encoded_value)
                else:
                    # For numerical features, use 0 as default
                    st.warning(f"⚠️ Missing value for feature '{feature_name}', using default value 0")
                    features.append(0.0)
                continue
            
            # Encode categorical features
            if feature_name in encoders:
                try:
                    encoded_value = encoders[feature_name].transform([str(value)])[0]
                    features.append(encoded_value)
                except (ValueError, KeyError) as e:
                    # If value not in encoder, use first class
                    st.warning(f"⚠️ Unknown value '{value}' for feature '{feature_name}', using default")
                    encoded_value = encoders[feature_name].transform([encoders[feature_name].classes_[0]])[0]
                    features.append(encoded_value)
            else:
                # Numerical feature - handle conversion safely
                try:
                    features.append(float(value))
                except (ValueError, TypeError) as e:
                    st.error(f"❌ Invalid value '{value}' for numerical feature '{feature_name}': {str(e)}")
                    features.append(0.0)
        
        # Convert to numpy array and reshape
        features_array = np.array(features).reshape(1, -1)
        
        # Scale features
        features_scaled = scaler.transform(features_array)
        
        # Make prediction
        prediction = model.predict(features_scaled)[0]
        prediction_proba = model.predict_proba(features_scaled)[0]
        
        # Get confidence (probability of predicted class)
        confidence = prediction_proba[prediction] * 100
        
        # Display results
        st.markdown("---")
        st.header("🎯 Prediction Results")
        
        # Create columns for results
        result_col1, result_col2 = st.columns(2)
        
        with result_col1:
            if prediction == 1:
                st.success(f"""
                ## ✅ Loan Approved
                
                **Confidence:** {confidence:.2f}%
                
                Congratulations! The model predicts that your loan application 
                is likely to be approved based on the provided information.
                """)
            else:
                st.error(f"""
                ## ❌ Loan Not Approved
                
                **Confidence:** {confidence:.2f}%
                
                The model predicts that your loan application may not be approved 
                based on the provided information. Please review your application 
                or consult with a financial advisor.
                """)
        
        with result_col2:
            st.subheader("📊 Probability Breakdown")
            
            not_approved_prob = round(prediction_proba[0] * 100, 2)
            approved_prob = round(prediction_proba[1] * 100, 2)
            
            # Display probabilities
            st.metric("Not Approved Probability", f"{not_approved_prob}%")
            st.metric("Approved Probability", f"{approved_prob}%")
            
            # Progress bars
            st.progress(not_approved_prob / 100, text=f"Not Approved: {not_approved_prob}%")
            st.progress(approved_prob / 100, text=f"Approved: {approved_prob}%")
        
        # Display input summary
        with st.expander("📋 View Input Summary"):
            st.json(feature_dict)
        
        # Calculate and display loan-to-income ratio
        total_income = applicant_income + coapplicant_income
        if total_income > 0:
            loan_to_income_ratio = (loan_amount / total_income) * 100
            st.info(f"💡 **Loan-to-Income Ratio:** {loan_to_income_ratio:.2f}%")
    
    except Exception as e:
        st.error(f"❌ An error occurred during prediction: {str(e)}")
        st.exception(e)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; padding: 1rem;'>
    <p>💰 Loan Approval Prediction System | Powered by KNN Machine Learning</p>
    <p><small>For demonstration purposes only. Not a substitute for professional financial advice.</small></p>
</div>
""", unsafe_allow_html=True)

