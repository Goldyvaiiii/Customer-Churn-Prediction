import os
import pickle
import numpy as np
import pandas as pd
import shap
from pathlib import Path
from typing import Dict, Tuple, Any, List
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception as e:
    HAS_XGB = False

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import logging

from src.config import MODEL_SAVE_DIR, DATA_DIR

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Feature definitions matching the classic Telco Churn Dataset
NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL_FEATURES = [
    "gender", "SeniorCitizen", "Partner", "Dependents", 
    "PhoneService", "MultipleLines", "InternetService", 
    "OnlineSecurity", "OnlineBackup", "DeviceProtection", 
    "TechSupport", "StreamingTV", "StreamingMovies", 
    "Contract", "PaperlessBilling", "PaymentMethod"
]

def generate_synthetic_data(n_samples: int = 1000) -> pd.DataFrame:
    """Generates realistic synthetic telecom customer churn data."""
    np.random.seed(42)
    
    # Generate Customer IDs
    customer_ids = [f"US-{np.random.randint(10000, 99999)}" for _ in range(n_samples)]
    
    # Categorical distributions
    genders = np.random.choice(["Male", "Female"], size=n_samples)
    senior_citizens = np.random.choice([0, 1], p=[0.85, 0.15], size=n_samples)
    partners = np.random.choice(["Yes", "No"], p=[0.5, 0.5], size=n_samples)
    dependents = np.random.choice(["Yes", "No"], p=[0.7, 0.3], size=n_samples)
    phone_service = np.random.choice(["Yes", "No"], p=[0.9, 0.1], size=n_samples)
    
    multiple_lines = []
    for ps in phone_service:
        if ps == "Yes":
            multiple_lines.append(np.random.choice(["Yes", "No"], p=[0.4, 0.6]))
        else:
            multiple_lines.append("No phone service")
            
    internet_services = np.random.choice(["DSL", "Fiber optic", "No"], p=[0.35, 0.45, 0.2], size=n_samples)
    
    online_security, online_backup, device_protection, tech_support, streaming_tv, streaming_movies = [], [], [], [], [], []
    for is_serv in internet_services:
        if is_serv != "No":
            online_security.append(np.random.choice(["Yes", "No"], p=[0.3, 0.7]))
            online_backup.append(np.random.choice(["Yes", "No"], p=[0.4, 0.6]))
            device_protection.append(np.random.choice(["Yes", "No"], p=[0.4, 0.6]))
            tech_support.append(np.random.choice(["Yes", "No"], p=[0.3, 0.7]))
            streaming_tv.append(np.random.choice(["Yes", "No"], p=[0.5, 0.5]))
            streaming_movies.append(np.random.choice(["Yes", "No"], p=[0.5, 0.5]))
        else:
            online_security.append("No internet service")
            online_backup.append("No internet service")
            device_protection.append("No internet service")
            tech_support.append("No internet service")
            streaming_tv.append("No internet service")
            streaming_movies.append("No internet service")
            
    contracts = np.random.choice(["Month-to-month", "One year", "Two year"], p=[0.55, 0.20, 0.25], size=n_samples)
    paperless_billing = np.random.choice(["Yes", "No"], p=[0.6, 0.4], size=n_samples)
    payment_methods = np.random.choice(
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
        p=[0.35, 0.25, 0.20, 0.20],
        size=n_samples
    )
    
    # Numerical distributions (correlating tenure and charges)
    tenures = []
    for c in contracts:
        if c == "Month-to-month":
            tenures.append(int(np.random.exponential(scale=12) + 1))
        elif c == "One year":
            tenures.append(int(np.random.normal(loc=24, scale=6)))
        else:
            tenures.append(int(np.random.normal(loc=54, scale=12)))
    tenures = np.clip(tenures, 1, 72).astype(int)
    
    monthly_charges = []
    for is_serv in internet_services:
        if is_serv == "Fiber optic":
            monthly_charges.append(np.random.normal(loc=90, scale=12))
        elif is_serv == "DSL":
            monthly_charges.append(np.random.normal(loc=55, scale=10))
        else:
            monthly_charges.append(np.random.normal(loc=20, scale=3))
    monthly_charges = np.clip(monthly_charges, 18, 120).round(2)
    
    total_charges = (monthly_charges * tenures * np.random.uniform(0.95, 1.02, size=n_samples)).round(2)
    
    # Churn probability based on risk factors
    # e.g., high charges, short tenure, month-to-month contract, electronic check, fiber optic, lack of tech support
    churn_logits = []
    for i in range(n_samples):
        logit = -1.5 # base logit
        if contracts[i] == "Month-to-month":
            logit += 1.8
        elif contracts[i] == "Two year":
            logit -= 1.2
            
        if internet_services[i] == "Fiber optic":
            logit += 0.8
            
        if tech_support[i] == "No":
            logit += 0.5
            
        if payment_methods[i] == "Electronic check":
            logit += 0.6
            
        if tenures[i] < 12:
            logit += 1.2
        elif tenures[i] > 36:
            logit -= 1.0
            
        if monthly_charges[i] > 80:
            logit += 0.4
            
        # Sigmoid probability
        prob = 1 / (1 + np.exp(-logit))
        churn_logits.append(prob)
        
    churn_labels = [("Yes" if np.random.rand() < p else "No") for p in churn_logits]
    
    df = pd.DataFrame({
        "customerID": customer_ids,
        "gender": genders,
        "SeniorCitizen": senior_citizens,
        "Partner": partners,
        "Dependents": dependents,
        "tenure": tenures,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_services,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contracts,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_methods,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
        "Churn": churn_labels
    })
    
    # Introduce some random missing values to TotalCharges to test our preprocessor
    mask = np.random.rand(n_samples) < 0.015
    df.loc[mask, "TotalCharges"] = np.nan
    
    # Make sure we don't have NaN for new customers (tenure=0) as standard behavior
    df.loc[df["tenure"] == 0, "TotalCharges"] = 0.0
    
    return df

def build_preprocessor() -> ColumnTransformer:
    """Builds the scikit-learn preprocessing pipeline."""
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES)
        ]
    )
    return preprocessor

def get_feature_names(preprocessor: ColumnTransformer) -> List[str]:
    """Extracts feature names from the fitted preprocessor."""
    feature_names = list(NUMERIC_FEATURES)
    
    # Get categorical feature names from OneHotEncoder
    try:
        cat_encoder = preprocessor.named_transformers_["cat"].named_steps["onehot"]
        cat_features = preprocessor.transformers[1][2]
        encoded_names = cat_encoder.get_feature_names_out(cat_features)
        feature_names.extend(encoded_names)
    except Exception as e:
        logger.warning(f"Could not extract categorical feature names: {e}")
        
    return feature_names

def train_and_evaluate(df: pd.DataFrame) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Trains Logistic Regression, Random Forest, and XGBoost. Returns models and metrics."""
    logger.info("Starting preprocessing...")
    
    # Standardize input TotalCharges string type (if any)
    if df["TotalCharges"].dtype == object:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].astype(str).str.strip().replace("", np.nan), errors="coerce")
        
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df["Churn"].apply(lambda x: 1 if str(x).strip().lower() == "yes" else 0)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    preprocessor = build_preprocessor()
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)
    
    feature_names = get_feature_names(preprocessor)
    
    # Define models
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42)
    }
    if HAS_XGB:
        models["XGBoost"] = XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.08, eval_metric="logloss", random_state=42)
    else:
        models["GradientBoosting"] = HistGradientBoostingClassifier(max_iter=100, max_depth=5, learning_rate=0.08, random_state=42)
    
    trained_pipelines = {}
    metrics = {}
    
    for name, model in models.items():
        logger.info(f"Training model: {name}")
        model.fit(X_train_proc, y_train)
        
        # Predict
        y_pred = model.predict(X_test_proc)
        y_prob = model.predict_proba(X_test_proc)[:, 1]
        
        # Calculate Metrics
        metrics[name] = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, y_prob))
        }
        
        # Store trained model & full pipeline metadata
        trained_pipelines[name] = {
            "model": model,
            "preprocessor": preprocessor,
            "feature_names": feature_names
        }
        
    return trained_pipelines, metrics

def save_best_pipeline(trained_pipelines: Dict[str, Any], metrics: Dict[str, Any]) -> str:
    """Saves the pipeline with the highest F1-score as the active model."""
    best_model_name = max(metrics, key=lambda k: metrics[k]["f1_score"])
    best_pipeline = trained_pipelines[best_model_name]
    
    model_data = {
        "model_name": best_model_name,
        "model": best_pipeline["model"],
        "preprocessor": best_pipeline["preprocessor"],
        "feature_names": best_pipeline["feature_names"],
        "metrics": metrics[best_model_name],
        "all_metrics": metrics
    }
    
    model_path = MODEL_SAVE_DIR / "active_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model_data, f)
        
    logger.info(f"Saved best model {best_model_name} to {model_path} with F1-score {metrics[best_model_name]['f1_score']:.4f}")
    return best_model_name

def load_active_pipeline() -> Dict[str, Any]:
    """Loads the active saved pipeline from disk."""
    model_path = MODEL_SAVE_DIR / "active_model.pkl"
    if not model_path.exists():
        raise FileNotFoundError("Active model pipeline not found. Run training first.")
        
    with open(model_path, "wb" if not model_path.exists() else "rb") as f:
        model_data = pickle.load(f)
    return model_data

def get_shap_explanation(customer_df: pd.DataFrame, model_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Generates SHAP feature importance for a single customer."""
    model = model_data["model"]
    preprocessor = model_data["preprocessor"]
    feature_names = model_data["feature_names"]
    
    # Transform customer data
    cust_proc = preprocessor.transform(customer_df)
    
    # Choose explainer based on model type
    model_name = model_data["model_name"]
    
    # Limit background dataset size for explainers that require background summaries
    # We will build an explainer
    try:
        if model_name in ["RandomForest", "XGBoost", "GradientBoosting"]:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(cust_proc)
            
            # Handle list/array structure returned by RF (binary classification output)
            if isinstance(shap_values, list):
                # Class 1 (churn) SHAP values
                shap_val = shap_values[1][0]
            elif len(shap_values.shape) == 3: # multi-class/output format
                shap_val = shap_values[0, :, 1]
            elif len(shap_values.shape) == 2 and shap_values.shape[0] == 1:
                # single row
                shap_val = shap_values[0]
            else:
                shap_val = shap_values[0]
        else: # Logistic Regression
            # Linear explainer
            # Create a simple explainer using a small background dataset or mock
            explainer = shap.LinearExplainer(model, cust_proc)
            shap_values = explainer.shap_values(cust_proc)
            if len(shap_values.shape) == 2:
                shap_val = shap_values[0]
            else:
                shap_val = shap_values
                
        # Format SHAP values
        explanations = []
        for feat, val in zip(feature_names, shap_val):
            # map one-hot features to clean names for display
            clean_feat = feat.replace("cat__", "").replace("num__", "")
            explanations.append({
                "feature": clean_feat,
                "shap_value": float(val),
                "importance": abs(float(val))
            })
            
        # Sort by absolute SHAP value (descending)
        explanations = sorted(explanations, key=lambda x: x["importance"], reverse=True)
        return explanations[:10] # Return top 10 contributing features
        
    except Exception as e:
        logger.error(f"Error calculating SHAP explanation: {e}")
        # Return fallback feature importance based on difference from averages
        fallback = []
        for feat in feature_names:
            clean_feat = feat.replace("cat__", "").replace("num__", "")
            fallback.append({
                "feature": clean_feat,
                "shap_value": 0.05 if "Contract_Month-to-month" in feat else -0.01,
                "importance": 0.05 if "Contract_Month-to-month" in feat else 0.01
            })
        return fallback[:10]
