"""
Loan Approval using KNN - Model Training Script
This script trains a KNN classifier for loan approval prediction.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

# Set style for plots
try:
    plt.style.use('seaborn-v0_8')
except OSError:
    try:
        plt.style.use('seaborn')
    except OSError:
        plt.style.use('default')
sns.set_palette("husl")

def load_data():
    """
    Load loan approval dataset. If dataset file doesn't exist, create synthetic data.
    """
    # Try multiple possible paths
    possible_paths = ['data.csv', 'loan_approval/data.csv', 'loan_data.csv']
    dataset_path = None
    for path in possible_paths:
        if os.path.exists(path):
            dataset_path = path
            break
    
    if dataset_path is None:
        dataset_path = 'data.csv'  # Default path
    
    if os.path.exists(dataset_path):
        df = pd.read_csv(dataset_path)
        print(f"Dataset loaded from {dataset_path}")
        print(f"Dataset shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        # Handle the real dataset structure
        # The real dataset uses 'loan_status' as target (0=Not Approved, 1=Approved)
        if 'loan_status' in df.columns:
            # Rename to match expected format
            df = df.rename(columns={'loan_status': 'Loan_Status'})
            # Convert to Y/N format if needed (0->N, 1->Y)
            if df['Loan_Status'].dtype in [int, float]:
                df['Loan_Status'] = df['Loan_Status'].map({0: 'N', 1: 'Y'})
    else:
        print("Dataset file not found. Creating synthetic loan approval dataset...")
        # Create synthetic loan approval dataset
        np.random.seed(42)
        n_samples = 1000
        
        # Features for loan approval
        data = {
            'Gender': np.random.choice(['Male', 'Female'], n_samples),
            'Married': np.random.choice(['Yes', 'No'], n_samples),
            'Dependents': np.random.choice(['0', '1', '2', '3+'], n_samples),
            'Education': np.random.choice(['Graduate', 'Not Graduate'], n_samples),
            'Self_Employed': np.random.choice(['Yes', 'No'], n_samples),
            'ApplicantIncome': np.random.randint(1500, 81000, n_samples),
            'CoapplicantIncome': np.random.randint(0, 41667, n_samples),
            'LoanAmount': np.random.randint(9, 700, n_samples),
            'Loan_Amount_Term': np.random.choice([12, 36, 60, 84, 120, 180, 240, 300, 360], n_samples),
            'Credit_History': np.random.choice([0, 1], n_samples, p=[0.2, 0.8]),
            'Property_Area': np.random.choice(['Urban', 'Semiurban', 'Rural'], n_samples)
        }
        
        df = pd.DataFrame(data)
        
        # Create target variable (Loan_Status: Y=1, N=0)
        # Higher income, good credit history, lower loan amount increase approval probability
        total_income = df['ApplicantIncome'] + df['CoapplicantIncome']
        loan_to_income_ratio = df['LoanAmount'] / (total_income + 1)
        
        approval_prob = (
            (df['Credit_History'] == 1).astype(int) * 0.4 +
            (loan_to_income_ratio < 0.3).astype(int) * 0.3 +
            (total_income > 5000).astype(int) * 0.15 +
            (df['Education'] == 'Graduate').astype(int) * 0.1 +
            (df['Property_Area'] == 'Urban').astype(int) * 0.05
        )
        df['Loan_Status'] = (approval_prob + np.random.random(n_samples) * 0.2 > 0.5).astype(int)
        df['Loan_Status'] = df['Loan_Status'].map({0: 'N', 1: 'Y'})
        
        # Save synthetic dataset
        os.makedirs('loan_approval', exist_ok=True)
        df.to_csv(dataset_path, index=False)
        print(f"Synthetic dataset created and saved to {dataset_path}")
    
    return df

def explore_data(df):
    """Perform data exploration and visualization."""
    print("\n" + "="*60)
    print("DATA EXPLORATION")
    print("="*60)
    
    print(f"\nDataset Shape: {df.shape}")
    print(f"\nColumn Names: {list(df.columns)}")
    print(f"\nData Types:\n{df.dtypes}")
    print(f"\nMissing Values:\n{df.isnull().sum()}")
    print(f"\nStatistical Summary:\n{df.describe()}")
    print(f"\nTarget Distribution:\n{df['Loan_Status'].value_counts()}")
    print(f"\nTarget Percentage:\n{df['Loan_Status'].value_counts(normalize=True) * 100}")
    
    # Create visualizations directory
    os.makedirs('loan_approval/visualizations', exist_ok=True)
    
    # Class distribution
    plt.figure(figsize=(8, 6))
    df['Loan_Status'].value_counts().plot(kind='bar', color=['#ff6b6b', '#51cf66'])
    plt.title('Loan Status Distribution', fontsize=14, pad=15)
    plt.xlabel('Loan Status (N: Not Approved, Y: Approved)', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig('loan_approval/visualizations/class_distribution.png', dpi=300, bbox_inches='tight')
    print("\nClass distribution plot saved to visualizations/class_distribution.png")
    plt.close()
    
    # Distribution plots for numerical features
    # Check which numerical features exist in the dataset
    if 'person_income' in df.columns:
        # Real dataset structure
        numerical_features = ['person_age', 'person_income', 'person_emp_exp', 'loan_amnt', 
                            'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length', 'credit_score']
        numerical_features = [f for f in numerical_features if f in df.columns]
    else:
        # Synthetic dataset structure
        numerical_features = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']
    
    # Dynamically create subplots based on number of features
    n_features = len(numerical_features)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    if n_features == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for idx, feature in enumerate(numerical_features):
        df[feature].hist(bins=30, ax=axes[idx], edgecolor='black')
        axes[idx].set_title(f'Distribution of {feature}')
        axes[idx].set_xlabel(feature)
        axes[idx].set_ylabel('Frequency')
    
    # Hide extra subplots
    for idx in range(n_features, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig('loan_approval/visualizations/numerical_distributions.png', dpi=300, bbox_inches='tight')
    print("Numerical feature distributions saved to visualizations/numerical_distributions.png")
    plt.close()
    
    # Categorical features analysis
    # Check which categorical features exist in the dataset
    if 'person_gender' in df.columns:
        # Real dataset structure
        categorical_features = ['person_gender', 'person_education', 'person_home_ownership', 
                              'loan_intent', 'previous_loan_defaults_on_file']
        categorical_features = [f for f in categorical_features if f in df.columns]
    else:
        # Synthetic dataset structure
        categorical_features = ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area', 'Credit_History']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for idx, feature in enumerate(categorical_features):
        if feature in df.columns:
            df[feature].value_counts().plot(kind='bar', ax=axes[idx], color='skyblue')
            axes[idx].set_title(f'{feature} Distribution')
            axes[idx].set_xlabel(feature)
            axes[idx].set_ylabel('Count')
            axes[idx].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('loan_approval/visualizations/categorical_distributions.png', dpi=300, bbox_inches='tight')
    print("Categorical feature distributions saved to visualizations/categorical_distributions.png")
    plt.close()
    
    # Correlation heatmap for numerical features
    numerical_df = df[numerical_features + ['Loan_Status']].copy()
    # Convert Loan_Status to numeric for correlation
    if numerical_df['Loan_Status'].dtype == 'object':
        numerical_df['Loan_Status'] = numerical_df['Loan_Status'].map({'N': 0, 'Y': 1})
    else:
        numerical_df['Loan_Status'] = numerical_df['Loan_Status'].astype(int)
    
    plt.figure(figsize=(10, 8))
    correlation_matrix = numerical_df.corr()
    sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    plt.title('Feature Correlation Heatmap', fontsize=16, pad=20)
    plt.tight_layout()
    plt.savefig('loan_approval/visualizations/correlation_heatmap.png', dpi=300, bbox_inches='tight')
    print("Correlation heatmap saved to visualizations/correlation_heatmap.png")
    plt.close()
    
    # Box plots for numerical features by loan status (limit to first 6 for readability)
    plot_features = numerical_features[:6]  # Limit to 6 features
    n_plot_features = len(plot_features)
    n_cols = 3
    n_rows = (n_plot_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    if n_plot_features == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for idx, feature in enumerate(plot_features):
        df.boxplot(column=feature, by='Loan_Status', ax=axes[idx])
        axes[idx].set_title(f'{feature} by Loan Status')
        axes[idx].set_xlabel('Loan Status')
        axes[idx].set_ylabel(feature)
    
    # Hide extra subplots
    for idx in range(n_plot_features, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('Feature Distributions by Loan Status', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig('loan_approval/visualizations/boxplots_by_status.png', dpi=300, bbox_inches='tight')
    print("Box plots saved to visualizations/boxplots_by_status.png")
    plt.close()

def prepare_data(df):
    """Prepare data for training."""
    # Create a copy to avoid modifying original
    df_processed = df.copy()
    
    # Encode target variable
    if df_processed['Loan_Status'].dtype == 'object':
        df_processed['Loan_Status'] = df_processed['Loan_Status'].map({'N': 0, 'Y': 1})
    else:
        df_processed['Loan_Status'] = df_processed['Loan_Status'].astype(int)
    
    # Handle missing values - fill with mode for categorical, median for numerical
    for col in df_processed.columns:
        if col != 'Loan_Status':
            if df_processed[col].dtype == 'object':
                df_processed[col] = df_processed[col].fillna(df_processed[col].mode()[0] if len(df_processed[col].mode()) > 0 else 'Unknown')
            else:
                df_processed[col] = df_processed[col].fillna(df_processed[col].median())
    
    # Separate features and target
    X = df_processed.drop('Loan_Status', axis=1)
    y = df_processed['Loan_Status']
    
    # Handle categorical variables
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    encoders = {}
    
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        encoders[col] = le
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, encoders, X.columns

def train_knn_model(X_train, X_test, y_train, y_test):
    """Train KNN model with hyperparameter tuning."""
    print("\n" + "="*60)
    print("MODEL TRAINING")
    print("="*60)
    
    # Hyperparameter tuning
    print("\nPerforming hyperparameter tuning...")
    param_grid = {'n_neighbors': range(1, 31)}
    knn = KNeighborsClassifier()
    grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    best_k = grid_search.best_params_['n_neighbors']
    print(f"\nBest k value: {best_k}")
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    
    # Train final model with best k
    best_model = KNeighborsClassifier(n_neighbors=best_k)
    best_model.fit(X_train, y_train)
    
    # Cross-validation scores
    cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='accuracy')
    print(f"\nCross-validation scores: {cv_scores}")
    print(f"Mean CV accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Plot accuracy vs k values
    k_values = range(1, 31)
    k_scores = []
    for k in k_values:
        knn_temp = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(knn_temp, X_train, y_train, cv=5, scoring='accuracy')
        k_scores.append(scores.mean())
    
    plt.figure(figsize=(12, 6))
    plt.plot(k_values, k_scores, marker='o', linestyle='-', linewidth=2, markersize=8)
    plt.axvline(x=best_k, color='r', linestyle='--', label=f'Best k={best_k}')
    plt.xlabel('k (Number of Neighbors)', fontsize=12)
    plt.ylabel('Cross-Validation Accuracy', fontsize=12)
    plt.title('KNN: Accuracy vs k Value', fontsize=14, pad=15)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('loan_approval/visualizations/knn_accuracy_vs_k.png', dpi=300, bbox_inches='tight')
    print("\nAccuracy vs k plot saved to visualizations/knn_accuracy_vs_k.png")
    plt.close()
    
    return best_model

def evaluate_model(model, X_test, y_test):
    """Evaluate the trained model."""
    print("\n" + "="*60)
    print("MODEL EVALUATION")
    print("="*60)
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"\nAccuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\nConfusion Matrix:\n{cm}")
    
    # Classification Report
    print(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")
    
    # Visualize confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                xticklabels=['Not Approved', 'Approved'],
                yticklabels=['Not Approved', 'Approved'])
    plt.title('Confusion Matrix', fontsize=14, pad=15)
    plt.ylabel('Actual', fontsize=12)
    plt.xlabel('Predicted', fontsize=12)
    plt.tight_layout()
    plt.savefig('loan_approval/visualizations/confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("\nConfusion matrix saved to visualizations/confusion_matrix.png")
    plt.close()
    
    return accuracy, precision, recall, f1

def save_model(model, scaler, encoders, feature_names):
    """Save the trained model, scaler, and encoders."""
    os.makedirs('loan_approval/models', exist_ok=True)
    
    joblib.dump(model, 'loan_approval/models/knn_model.pkl')
    joblib.dump(scaler, 'loan_approval/models/scaler.pkl')
    joblib.dump(encoders, 'loan_approval/models/encoders.pkl')
    joblib.dump(feature_names, 'loan_approval/models/feature_names.pkl')
    
    print("\n" + "="*60)
    print("MODEL SAVED")
    print("="*60)
    print("\nModel saved to: loan_approval/models/knn_model.pkl")
    print("Scaler saved to: loan_approval/models/scaler.pkl")
    print("Encoders saved to: loan_approval/models/encoders.pkl")
    print("Feature names saved to: loan_approval/models/feature_names.pkl")

def main():
    """Main function to run the training pipeline."""
    print("="*60)
    print("LOAN APPROVAL USING KNN - MODEL TRAINING")
    print("="*60)
    
    # Load data
    df = load_data()
    
    # Explore data
    explore_data(df)
    
    # Prepare data
    X_train, X_test, y_train, y_test, scaler, encoders, feature_names = prepare_data(df)
    
    # Train model
    model = train_knn_model(X_train, X_test, y_train, y_test)
    
    # Evaluate model
    evaluate_model(model, X_test, y_test)
    
    # Save model
    save_model(model, scaler, encoders, feature_names)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)

if __name__ == "__main__":
    main()

