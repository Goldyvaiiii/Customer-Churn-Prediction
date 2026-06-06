"""
utils.py — Unified backend utility layer for the Streamlit frontend.

Supports two modes:
  1. STANDALONE mode: Runs ML pipeline directly inside Streamlit (for Streamlit Cloud)
  2. API mode: Delegates all calls to the FastAPI backend (for local/Docker usage)

Mode is auto-detected: if FastAPI is reachable → API mode; otherwise → Standalone.
"""
import os
import sys
import logging
import sqlite3
import joblib
import json
import pickle
import io
from pathlib import Path
from typing import Dict, Any, List, Optional

import requests
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# ─── Path resolution ────────────────────────────────────────────────────
# Works whether running from repo root or from src/frontend/
_HERE = Path(__file__).resolve().parent           # src/frontend/
_SRC  = _HERE.parent                              # src/
_ROOT = _SRC.parent                               # project root

# Ensure src/ is importable
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# API config
API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")

# ─── Mode detection ──────────────────────────────────────────────────────
def check_backend_health() -> bool:
    """Returns True if the FastAPI backend is reachable."""
    try:
        r = requests.get(f"{API_BASE_URL}/", timeout=2)
        return r.status_code == 200
    except Exception:
        return False

def _is_api_mode() -> bool:
    return check_backend_health()

# ─── Standalone helpers ──────────────────────────────────────────────────
def _get_db_path() -> Path:
    """Returns the SQLite database path."""
    candidates = [
        _ROOT / "database.sqlite",
        _ROOT / "churn.db",
        Path("database.sqlite"),
    ]
    for p in candidates:
        if p.exists():
            return p
    # Return default location even if it doesn't exist yet
    return _ROOT / "database.sqlite"

def _get_model_path() -> Optional[Path]:
    """Finds the saved model file."""
    candidates = [
        _SRC / "ml" / "models" / "active_model.pkl",
        _SRC / "ml" / "models" / "best_model.pkl",
        _ROOT / "models" / "best_model.pkl",
        _ROOT / "best_model.pkl",
        _SRC / "ml" / "best_model.pkl",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None

def _load_standalone_model():
    """Loads the trained model from disk."""
    model_path = _get_model_path()
    if model_path:
        return joblib.load(model_path)
    return None

def _standalone_get_customers() -> List[Dict[str, Any]]:
    """Reads customers directly from SQLite joined with predictions."""
    db = _get_db_path()
    if not db.exists():
        return []
    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    try:
        query = """
            SELECT 
                c.*,
                p.churn_probability,
                p.churn_prediction,
                p.risk_category,
                p.model_version
            FROM customers c
            LEFT OUTER JOIN predictions p ON c.customer_id = p.customer_id
        """
        rows = cur.execute(query).fetchall()
        output = []
        for r in rows:
            d = dict(r)
            if d.get("risk_category") is None:
                d["churn_probability"] = None
                d["churn_prediction"] = None
                d["risk_category"] = "Unpredicted"
                d["model_version"] = "None"
            output.append(d)
        return output
    except Exception as e:
        logger.error(f"DB read error: {e}")
        return []
    finally:
        conn.close()

def _standalone_get_summary() -> Dict[str, Any]:
    """Computes the executive KPI summary from SQLite."""
    customers = _standalone_get_customers()
    if not customers:
        return {
            "total_customers": 0, "churned_customers": 0,
            "predicted_churn_rate": 0.0, "actual_churn_rate": 0.0,
            "revenue_at_risk": 0.0, "average_tenure": 0.0,
            "risk_distribution": {"Low": 0, "Medium": 0, "High": 0}
        }
    df = pd.DataFrame(customers)
    total = len(df)
    churned = int((df["churn"] == "Yes").sum()) if "churn" in df.columns else 0
    actual_rate = churned / total if total > 0 else 0.0

    predicted_rate = 0.0
    revenue_at_risk = 0.0
    risk_dist = {"Low": 0, "Medium": 0, "High": 0}

    if "churn_probability" in df.columns and df["churn_probability"].notna().any():
        probs = pd.to_numeric(df["churn_probability"], errors="coerce").fillna(0)
        predicted_rate = float((probs > 0.5).sum() / total)
        revenue_at_risk = float((df.loc[probs > 0.5, "monthly_charges"]
                                  .apply(pd.to_numeric, errors="coerce")
                                  .fillna(0).sum()))

    if "risk_category" in df.columns:
        counts = df["risk_category"].value_counts()
        for k in ["Low", "Medium", "High"]:
            risk_dist[k] = int(counts.get(k, 0))

    avg_tenure = float(pd.to_numeric(df.get("tenure", pd.Series([])),
                                     errors="coerce").fillna(0).mean())

    return {
        "total_customers": total,
        "churned_customers": churned,
        "predicted_churn_rate": predicted_rate,
        "actual_churn_rate": actual_rate,
        "revenue_at_risk": revenue_at_risk,
        "average_tenure": round(avg_tenure, 2),
        "risk_distribution": risk_dist
    }

def _standalone_get_audit_logs() -> List[Dict[str, Any]]:
    """Reads audit logs directly from SQLite."""
    db = _get_db_path()
    if not db.exists():
        return []
    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    try:
        rows = cur.execute(
            "SELECT * FROM audit_logs ORDER BY timestamp DESC LIMIT 200"
        ).fetchall()
        return [dict(r) for r in rows]
    except Exception:
        return []
    finally:
        conn.close()

def _standalone_shap_explain(customer_id: str) -> Dict[str, Any]:
    """Runs SHAP explanation directly using loaded model."""
    try:
        from ml.pipeline import get_shap_explanation, NUMERIC_FEATURES, CATEGORICAL_FEATURES
    except ImportError:
        return {"error": "ML dependencies not available in this environment."}

    customers = _standalone_get_customers()
    if not customers:
        return {"error": "No customer data available."}

    df = pd.DataFrame(customers)
    row = df[df["customer_id"] == customer_id]
    if row.empty:
        return {"error": f"Customer {customer_id} not found."}

    model_bundle = _load_standalone_model()
    if model_bundle is None:
        return {"error": "No trained model found. Please train first."}

    # Map snake_case → PascalCase for the pipeline
    cust = row.iloc[0]
    c_dict = {
        "customerID": cust["customer_id"], "gender": cust["gender"],
        "SeniorCitizen": cust["senior_citizen"], "Partner": cust["partner"],
        "Dependents": cust["dependents"], "tenure": cust["tenure"],
        "PhoneService": cust["phone_service"], "MultipleLines": cust["multiple_lines"],
        "InternetService": cust["internet_service"], "OnlineSecurity": cust["online_security"],
        "OnlineBackup": cust["online_backup"], "DeviceProtection": cust["device_protection"],
        "TechSupport": cust["tech_support"], "StreamingTV": cust["streaming_tv"],
        "StreamingMovies": cust["streaming_movies"], "Contract": cust["contract"],
        "PaperlessBilling": cust["paperless_billing"], "PaymentMethod": cust["payment_method"],
        "MonthlyCharges": cust["monthly_charges"], "TotalCharges": cust["total_charges"],
        "Churn": cust.get("churn", "No")
    }
    c_df = pd.DataFrame([c_dict])

    try:
        factors = get_shap_explanation(c_df, model_bundle)
        
        preprocessor = model_bundle.get("preprocessor")
        model = model_bundle.get("model")
        features = NUMERIC_FEATURES + CATEGORICAL_FEATURES
        X = c_df[features]
        X_proc = preprocessor.transform(X)
        prob = float(model.predict_proba(X_proc)[0][1])
        risk = "High" if prob >= 0.7 else "Medium" if prob >= 0.3 else "Low"
        
        return {
            "customer_id": customer_id,
            "churn_probability": prob,
            "risk_category": risk,
            "top_factors": factors
        }
    except Exception as e:
        logger.error(f"SHAP explanation failed: {e}")
        return {"error": f"SHAP calculation error: {e}"}

def _standalone_rag_query(query: str, customer_id: Optional[str] = None) -> Dict[str, Any]:
    """Performs RAG retrieval directly using ChromaDB."""
    try:
        from rag.retriever import generate_retrieval_response
        result = generate_retrieval_response(query, customer_id=customer_id)
        return result
    except Exception as e:
        # Graceful fallback
        return {
            "query": query,
            "response": (
                "### 📋 Retention Playbook (Offline Fallback)\n\n"
                "The RAG knowledge base is temporarily unavailable in this environment.\n\n"
                "**General High-Risk Recommendations:**\n"
                "1. **Schedule a call** with your CSM within 24 hours.\n"
                "2. **Offer a 20% discount** on annual contract conversion.\n"
                "3. **Audit support logs** for unresolved complaints.\n"
                "4. **Propose a loyalty bundle** (e.g., streaming + security add-ons).\n\n"
                f"*Note: Full RAG not available — {str(e)[:100]}*"
            ),
            "mode": "Offline Fallback",
            "sources": []
        }

def _standalone_seed_and_train() -> Dict[str, Any]:
    """Runs ML training pipeline directly in Streamlit Cloud."""
    try:
        from ml.pipeline import (
            generate_synthetic_data, train_and_evaluate,
            save_best_pipeline, NUMERIC_FEATURES, CATEGORICAL_FEATURES
        )
        from database import SessionLocal, init_db, Customer as CustomerModel, Prediction as PredictionModel

        init_db()
        df = generate_synthetic_data(1000)

        # Write customers to DB
        db = SessionLocal()
        db.query(PredictionModel).delete()
        db.query(CustomerModel).delete()
        db.commit()

        for _, row in df.iterrows():
            cust = CustomerModel(
                customer_id=row["customerID"],
                gender=row["gender"],
                senior_citizen=int(row["SeniorCitizen"]),
                partner=row["Partner"],
                dependents=row["Dependents"],
                tenure=int(row["tenure"]),
                phone_service=row["PhoneService"],
                multiple_lines=row["MultipleLines"],
                internet_service=row["InternetService"],
                online_security=row["OnlineSecurity"],
                online_backup=row["OnlineBackup"],
                device_protection=row["DeviceProtection"],
                tech_support=row["TechSupport"],
                streaming_tv=row["StreamingTV"],
                streaming_movies=row["StreamingMovies"],
                contract=row["Contract"],
                paperless_billing=row["PaperlessBilling"],
                payment_method=row["PaymentMethod"],
                monthly_charges=float(row["MonthlyCharges"]),
                total_charges=float(row["TotalCharges"]) if not pd.isna(row["TotalCharges"]) else 0.0,
                churn=row["Churn"]
            )
            db.add(cust)
        db.commit()

        # Train
        trained_pipelines, metrics_dict = train_and_evaluate(df)
        best_name = save_best_pipeline(trained_pipelines, metrics_dict)
        
        # Load the best model bundle to get preprocessor and model
        best_pipeline = trained_pipelines[best_name]
        best_model = best_pipeline["model"]
        preprocessor = best_pipeline["preprocessor"]

        # Update predictions in DB
        all_custs = db.query(CustomerModel).all()
        for c in all_custs:
            try:
                c_dict = {
                    "customerID": c.customer_id, "gender": c.gender,
                    "SeniorCitizen": c.senior_citizen, "Partner": c.partner,
                    "Dependents": c.dependents, "tenure": c.tenure,
                    "PhoneService": c.phone_service, "MultipleLines": c.multiple_lines,
                    "InternetService": c.internet_service, "OnlineSecurity": c.online_security,
                    "OnlineBackup": c.online_backup, "DeviceProtection": c.device_protection,
                    "TechSupport": c.tech_support, "StreamingTV": c.streaming_tv,
                    "StreamingMovies": c.streaming_movies, "Contract": c.contract,
                    "PaperlessBilling": c.paperless_billing, "PaymentMethod": c.payment_method,
                    "MonthlyCharges": c.monthly_charges, "TotalCharges": c.total_charges,
                    "Churn": c.churn
                }
                features = NUMERIC_FEATURES + CATEGORICAL_FEATURES
                row_df = pd.DataFrame([c_dict])[features]
                X_t = preprocessor.transform(row_df)
                prob = float(best_model.predict_proba(X_t)[0][1])
                risk = "High" if prob >= 0.7 else "Medium" if prob >= 0.3 else "Low"
                pred_val = 1 if prob >= 0.5 else 0
                
                pred_obj = PredictionModel(
                    customer_id=c.customer_id,
                    churn_probability=prob,
                    churn_prediction=pred_val,
                    risk_category=risk,
                    explainability=json.dumps({"factors": []}),
                    model_version=best_name
                )
                db.add(pred_obj)
            except Exception as e:
                logger.error(f"Error predicting for {c.customer_id}: {e}")
        db.commit()
        db.close()

        metrics = metrics_dict[best_name]
        return {
            "status": "success",
            "best_model": best_name,
            "metrics": metrics,
            "all_metrics": metrics_dict
        }
    except Exception as e:
        raise RuntimeError(f"Standalone training failed: {e}")

def _standalone_upload_csv(file_bytes: bytes, filename: str) -> Dict[str, Any]:
    """Processes uploaded CSV and runs predictions in standalone mode."""
    try:
        from database import SessionLocal, init_db, Customer as CustomerModel, Prediction as PredictionModel
        from ml.pipeline import train_and_evaluate, save_best_pipeline, NUMERIC_FEATURES, CATEGORICAL_FEATURES
        import joblib

        init_db()
        try:
            df = pd.read_csv(io.BytesIO(file_bytes), encoding="utf-8")
        except UnicodeDecodeError:
            df = pd.read_csv(io.BytesIO(file_bytes), encoding="latin-1")

        # Normalize column names
        df.columns = [c.strip() for c in df.columns]

        db = SessionLocal()
        db.query(PredictionModel).delete()
        db.query(CustomerModel).delete()
        db.commit()

        # Determine column mapping
        def gc(opts):
            for o in opts:
                if o in df.columns:
                    return o
            return None

        count = 0
        customers_to_add = []
        for _, row in df.iterrows():
            try:
                cid_col = gc(["customerID", "CustomerID", "customer_id"])
                cust = CustomerModel(
                    customer_id=str(row[cid_col]) if cid_col else f"C-{count}",
                    gender=str(row.get(gc(["gender", "Gender"]), "Male")),
                    senior_citizen=int(row.get(gc(["SeniorCitizen", "senior_citizen"]), 0)),
                    partner=str(row.get(gc(["Partner", "partner"]), "No")),
                    dependents=str(row.get(gc(["Dependents", "dependents"]), "No")),
                    tenure=int(row.get(gc(["tenure", "Tenure"]), 0)),
                    phone_service=str(row.get(gc(["PhoneService", "phone_service"]), "Yes")),
                    multiple_lines=str(row.get(gc(["MultipleLines", "multiple_lines"]), "No")),
                    internet_service=str(row.get(gc(["InternetService", "internet_service"]), "Fiber optic")),
                    online_security=str(row.get(gc(["OnlineSecurity", "online_security"]), "No")),
                    online_backup=str(row.get(gc(["OnlineBackup", "online_backup"]), "No")),
                    device_protection=str(row.get(gc(["DeviceProtection", "device_protection"]), "No")),
                    tech_support=str(row.get(gc(["TechSupport", "tech_support"]), "No")),
                    streaming_tv=str(row.get(gc(["StreamingTV", "streaming_tv"]), "No")),
                    streaming_movies=str(row.get(gc(["StreamingMovies", "streaming_movies"]), "No")),
                    contract=str(row.get(gc(["Contract", "contract"]), "Month-to-month")),
                    paperless_billing=str(row.get(gc(["PaperlessBilling", "paperless_billing"]), "Yes")),
                    payment_method=str(row.get(gc(["PaymentMethod", "payment_method"]), "Electronic check")),
                    monthly_charges=float(row.get(gc(["MonthlyCharges", "monthly_charges"]), 0)),
                    total_charges=float(str(row.get(gc(["TotalCharges", "total_charges"]), 0)).replace(" ", "") or 0),
                    churn=str(row.get(gc(["Churn", "churn"]), "No")),
                )
                db.add(cust)
                customers_to_add.append(cust)
                count += 1
            except Exception:
                continue

        db.commit()

        # Load active model or train one if not exists
        model_bundle = _load_standalone_model()
        if not model_bundle:
            logger.info("No active model found. Training on uploaded CSV data.")
            if count >= 10:
                try:
                    trained_pipelines, metrics_dict = train_and_evaluate(df)
                    best_name = save_best_pipeline(trained_pipelines, metrics_dict)
                    model_bundle = trained_pipelines[best_name]
                    model_bundle["model_name"] = best_name
                except Exception as train_err:
                    logger.error(f"Failed to train on uploaded data: {train_err}")
            
        predictions_run = 0
        if model_bundle:
            logger.info("Running predictions on uploaded dataset...")
            preprocessor = model_bundle.get("preprocessor")
            model = model_bundle.get("model")
            model_name = model_bundle.get("model_name", "Model")
            
            cust_dicts = []
            for c in customers_to_add:
                cust_dicts.append({
                    "customerID": c.customer_id, "gender": c.gender,
                    "SeniorCitizen": c.senior_citizen, "Partner": c.partner,
                    "Dependents": c.dependents, "tenure": c.tenure,
                    "PhoneService": c.phone_service, "MultipleLines": c.multiple_lines,
                    "InternetService": c.internet_service, "OnlineSecurity": c.online_security,
                    "OnlineBackup": c.online_backup, "DeviceProtection": c.device_protection,
                    "TechSupport": c.tech_support, "StreamingTV": c.streaming_tv,
                    "StreamingMovies": c.streaming_movies, "Contract": c.contract,
                    "PaperlessBilling": c.paperless_billing, "PaymentMethod": c.payment_method,
                    "MonthlyCharges": c.monthly_charges, "TotalCharges": c.total_charges,
                    "Churn": c.churn
                })
            
            cust_df = pd.DataFrame(cust_dicts)
            features = NUMERIC_FEATURES + CATEGORICAL_FEATURES
            X_proc = preprocessor.transform(cust_df[features])
            probs = model.predict_proba(X_proc)[:, 1]
            preds = model.predict(X_proc)
            
            for c, prob, pred in zip(customers_to_add, probs, preds):
                prob_val = float(prob)
                pred_val = int(pred)
                risk = "High" if prob_val >= 0.7 else "Medium" if prob_val >= 0.3 else "Low"
                
                pred_obj = PredictionModel(
                    customer_id=c.customer_id,
                    churn_probability=prob_val,
                    churn_prediction=pred_val,
                    risk_category=risk,
                    explainability=json.dumps({"factors": []}),
                    model_version=model_name
                )
                db.add(pred_obj)
                predictions_run += 1
            db.commit()

        db.close()
        return {
            "status": "success",
            "records_count": count,
            "predictions_generated": predictions_run
        }
    except Exception as e:
        raise RuntimeError(f"CSV upload failed: {e}")


# ─── Public API (auto-routes to API or standalone) ────────────────────────

def upload_csv(file_bytes: bytes, filename: str) -> Dict[str, Any]:
    if _is_api_mode():
        files = {"file": (filename, file_bytes, "text/csv")}
        r = requests.post(f"{API_BASE_URL}/api/upload", files=files)
        r.raise_for_status()
        return r.json()
    return _standalone_upload_csv(file_bytes, filename)

def trigger_training() -> Dict[str, Any]:
    if _is_api_mode():
        r = requests.post(f"{API_BASE_URL}/api/train")
        r.raise_for_status()
        return r.json()
    return _standalone_seed_and_train()

def get_customers() -> List[Dict[str, Any]]:
    if _is_api_mode():
        r = requests.get(f"{API_BASE_URL}/api/customers")
        r.raise_for_status()
        return r.json()
    return _standalone_get_customers()

def get_executive_summary() -> Dict[str, Any]:
    if _is_api_mode():
        r = requests.get(f"{API_BASE_URL}/api/analytics/summary")
        r.raise_for_status()
        return r.json()
    return _standalone_get_summary()

def get_shap_explanation(customer_id: str) -> Dict[str, Any]:
    if _is_api_mode():
        r = requests.get(f"{API_BASE_URL}/api/explain/{customer_id}")
        r.raise_for_status()
        return r.json()
    return _standalone_shap_explain(customer_id)

def ask_rag_assistant(query: str, customer_id: Optional[str] = None) -> Dict[str, Any]:
    if _is_api_mode():
        r = requests.post(f"{API_BASE_URL}/api/rag/query",
                          json={"query": query, "customer_id": customer_id})
        r.raise_for_status()
        return r.json()
    return _standalone_rag_query(query, customer_id)

def get_audit_logs() -> List[Dict[str, Any]]:
    if _is_api_mode():
        r = requests.get(f"{API_BASE_URL}/api/audit-logs")
        r.raise_for_status()
        return r.json()
    return _standalone_get_audit_logs()
