# Loan Approval Streamlit App - Deployment Guide

This guide explains how to deploy the Loan Approval Streamlit application.

## 🚀 Quick Start

### Local Development

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Train the Model** (if not already done)
   ```bash
   python train_model.py
   ```
   This will create the required model files in `loan_approval/models/` directory:
   - `knn_model.pkl`
   - `scaler.pkl`
   - `encoders.pkl`
   - `feature_names.pkl`

3. **Run the Streamlit App**
   ```bash
   streamlit run streamlit_app.py
   ```

The app will be available at `http://localhost:8502`

## ☁️ Streamlit Cloud Deployment

### Option 1: Deploy from GitHub

1. **Push your code to GitHub**
   ```bash
   git add .
   git commit -m "Add Streamlit app for loan approval"
   git push origin main
   ```

2. **Go to [Streamlit Cloud](https://share.streamlit.io/)**
   - Sign in with your GitHub account
   - Click "New app"
   - Select your repository
   - Set the following:
     - **Main file path**: `loan_approval/streamlit_app.py`
     - **Branch**: `main` (or your default branch)
   - Click "Deploy!"

3. **Important**: Make sure your model files are included in the repository, or train them during deployment

### Option 2: Deploy with Model Training

If you want to train models automatically during deployment, create a setup script:

1. **Create `setup.sh`** (for Unix systems):
   ```bash
   #!/bin/bash
   python loan_approval/train_model.py
   ```

2. **Update `streamlit_app.py`** to check for models and train if missing

## 📁 Project Structure

```
loan_approval/
├── streamlit_app.py          # Main Streamlit application
├── train_model.py            # Model training script
├── requirements.txt          # Python dependencies
├── .streamlit/
│   └── config.toml          # Streamlit configuration
├── loan_approval/
│   ├── loan_data.csv        # Training dataset
│   └── models/              # Trained model files
│       ├── knn_model.pkl
│       ├── scaler.pkl
│       ├── encoders.pkl
│       └── feature_names.pkl
└── DEPLOYMENT.md            # This file
```

## 🔧 Configuration

The app is configured via `.streamlit/config.toml`. Key settings:

- **Theme**: Custom color scheme
- **Server**: Headless mode for deployment
- **Browser**: Usage stats disabled

## 📋 Requirements

See `requirements.txt` for all dependencies. Minimum requirements:

- streamlit >= 1.28.0
- pandas >= 2.1.3
- numpy >= 1.26.2
- scikit-learn >= 1.3.2
- joblib >= 1.3.2

## 🔍 Troubleshooting

### Model Not Found Error

If you see "Model Not Found", ensure:
1. Models are trained: `python train_model.py`
2. Model files exist in `loan_approval/models/` directory
3. Paths in `streamlit_app.py` match your directory structure

### Path Issues on Streamlit Cloud

The app tries multiple path configurations automatically. If models aren't loading:
1. Check the file structure matches what's expected
2. Ensure model files are committed to git
3. Check Streamlit Cloud logs for path errors

### Encoder Issues

If you encounter encoding errors:
1. Ensure `encoders.pkl` is created during training
2. Check that categorical features match training data
3. Verify encoder paths in the app

## 📝 Notes

- This is a demonstration system for educational purposes
- For actual loan decisions, consult financial institutions
- The app uses caching (`@st.cache_resource`) to load models efficiently

## 🌐 Alternative Deployment Options

### Docker Deployment

Create a `Dockerfile`:

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8502

CMD ["streamlit", "run", "loan_approval/streamlit_app.py", "--server.port=8502", "--server.address=0.0.0.0"]
```

Build and run:
```bash
docker build -t loan-approval-app .
docker run -p 8502:8502 loan-approval-app
```

### Other Platforms

- **Heroku**: Use the Streamlit buildpack
- **AWS EC2**: Run with `streamlit run` as a service
- **Google Cloud Run**: Deploy as a containerized app

