import io
import json
import logging
import pandas as pd
import numpy as np
from fastapi import FastAPI, Depends, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from typing import List, Dict, Any, Optional

from src.config import API_HOST, API_PORT
from src.database import init_db, get_db, Customer, Prediction, AuditLog
from src.ml.models import load_model, is_model_trained
from src.ml.pipeline import (
    train_and_evaluate, save_best_pipeline, generate_synthetic_data,
    get_shap_explanation, NUMERIC_FEATURES, CATEGORICAL_FEATURES
)
from src.rag.vectorstore import ingest_documents
from src.rag.retriever import generate_retrieval_response
from src.api.schemas import (
    RAGQueryRequest, RAGQueryResponse, TrainResponse, ExplainResponse, 
    ExecutiveSummary, CustomerResponse
)

# Initialize logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Create FastAPI app
app = FastAPI(
    title="Customer Churn Prediction API",
    description="Enterprise-grade Customer Churn Prediction and RAG Retention Assistant API Layer",
    version="1.0.0"
)

# Add CORS middleware for frontend connection
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def startup_event():
    """Initializes the SQLite database structure and ingests vector playbooks."""
    logger.info("Initializing SQLite database tables...")
    init_db()
    
    logger.info("Ingesting playbooks into ChromaDB...")
    try:
        ingest_documents()
    except Exception as e:
        logger.error(f"Failed to ingest documents on startup: {e}")

@app.get("/")
def read_root():
    return {
        "status": "online",
        "app": "Customer Churn Prediction API",
        "model_trained": is_model_trained()
    }

@app.post("/api/upload", response_model=Dict[str, Any])
def upload_dataset(file: UploadFile = File(...), db: Session = Depends(get_db)):
    """Uploads a customer CSV dataset, stores it in SQLite, and triggers auto-predictions if a model exists."""
    logger.info(f"Uploading file: {file.filename}")
    
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are allowed.")
        
    try:
        contents = file.file.read()
        df = pd.read_csv(io.BytesIO(contents))
        
        # Validate columns
        required_cols = ["customerID"] + NUMERIC_FEATURES + CATEGORICAL_FEATURES
        missing = [col for col in required_cols if col not in df.columns]
        
        # Tolerate missing "Churn" column for testing/predictions, but check other features
        if missing:
            raise HTTPException(
                status_code=400, 
                detail=f"Uploaded CSV is missing required columns: {missing}"
            )
            
        # Clear existing customer database to avoid duplicates / overlay new batch
        db.query(Prediction).delete()
        db.query(Customer).delete()
        db.commit()
        
        # Standardize total charges
        if "TotalCharges" in df.columns:
            df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].astype(str).str.strip().replace("", np.nan), errors="coerce")
            
        # Parse and save customers
        records_added = 0
        active_model = load_model()
        
        for _, row in df.iterrows():
            cust_id = str(row["customerID"]).strip()
            
            cust_obj = Customer(
                customer_id=cust_id,
                gender=str(row.get("gender", "Male")),
                senior_citizen=int(row.get("SeniorCitizen", 0)),
                partner=str(row.get("Partner", "No")),
                dependents=str(row.get("Dependents", "No")),
                tenure=int(row.get("tenure", 0)),
                phone_service=str(row.get("PhoneService", "Yes")),
                multiple_lines=str(row.get("MultipleLines", "No")),
                internet_service=str(row.get("InternetService", "DSL")),
                online_security=str(row.get("OnlineSecurity", "No")),
                online_backup=str(row.get("OnlineBackup", "No")),
                device_protection=str(row.get("DeviceProtection", "No")),
                tech_support=str(row.get("TechSupport", "No")),
                streaming_tv=str(row.get("StreamingTV", "No")),
                streaming_movies=str(row.get("StreamingMovies", "No")),
                contract=str(row.get("Contract", "Month-to-month")),
                paperless_billing=str(row.get("PaperlessBilling", "Yes")),
                payment_method=str(row.get("PaymentMethod", "Electronic check")),
                monthly_charges=float(row.get("MonthlyCharges", 0.0)),
                total_charges=float(row.get("TotalCharges", 0.0)) if not pd.isna(row.get("TotalCharges")) else 0.0,
                churn=str(row.get("Churn", "No"))
            )
            db.add(cust_obj)
            records_added += 1
            
        db.commit()
        
        # Log to Audit
        audit = AuditLog(
            action="UPLOAD_DATASET",
            details=f"Uploaded dataset {file.filename} with {records_added} customer records."
        )
        db.add(audit)
        db.commit()
        
        # Auto-predict if model exists
        predictions_run = 0
        if active_model:
            logger.info("Auto-running predictions on uploaded dataset...")
            # Query back all customers to ensure clean processing
            customers = db.query(Customer).all()
            
            cust_dicts = []
            for c in customers:
                d = {col.name: getattr(c, col.name) for col in c.__table__.columns}
                d["customerID"] = d["customer_id"] # pipeline expects customerID
                cust_dicts.append(d)
                
            cust_df = pd.DataFrame(cust_dicts)
            
            # Preprocess and predict
            preprocessor = active_model["preprocessor"]
            model = active_model["model"]
            model_name = active_model["model_name"]
            
            X_proc = preprocessor.transform(cust_df)
            probs = model.predict_proba(X_proc)[:, 1]
            preds = model.predict(X_proc)
            
            for c, prob, pred in zip(customers, probs, preds):
                prob_val = float(prob)
                pred_val = int(pred)
                
                # Assign Risk category
                if prob_val >= 0.7:
                    risk = "High"
                elif prob_val >= 0.3:
                    risk = "Medium"
                else:
                    risk = "Low"
                    
                # Get quick SHAP local explanation placeholder to save DB overhead
                # SHAP will be computed in real-time on request, but store general info
                exp_summary = json.dumps({"factors": ["Contract_Month-to-month" if c.contract == "Month-to-month" else "Charges"]})
                
                pred_obj = Prediction(
                    customer_id=c.customer_id,
                    churn_probability=prob_val,
                    churn_prediction=pred_val,
                    risk_category=risk,
                    explainability=exp_summary,
                    model_version=model_name
                )
                db.add(pred_obj)
                predictions_run += 1
                
            db.commit()
            logger.info(f"Auto-prediction completed for {predictions_run} customers.")
            
        return {
            "status": "success",
            "message": f"Successfully loaded {records_added} customer profiles.",
            "records_count": records_added,
            "predictions_generated": predictions_run
        }
    except Exception as e:
        db.rollback()
        logger.error(f"Error importing CSV dataset: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process CSV file: {str(e)}")

@app.post("/api/train", response_model=TrainResponse)
def train_model(db: Session = Depends(get_db)):
    """Trains the models on current database records. Generates synthetic records if DB is empty."""
    logger.info("Training request received...")
    
    # Query all customers
    customers = db.query(Customer).all()
    
    # Check if empty, if so, load synthetic dataset for training
    if len(customers) < 50:
        logger.info("Database is empty or too small. Seeding with 1000 synthetic customers for demo.")
        df = generate_synthetic_data(n_samples=1000)
        
        # Save synthetic data to database
        for _, row in df.iterrows():
            cust_obj = Customer(
                customer_id=str(row["customerID"]),
                gender=str(row.get("gender", "Male")),
                senior_citizen=int(row.get("SeniorCitizen", 0)),
                partner=str(row.get("Partner", "No")),
                dependents=str(row.get("Dependents", "No")),
                tenure=int(row.get("tenure", 0)),
                phone_service=str(row.get("PhoneService", "Yes")),
                multiple_lines=str(row.get("MultipleLines", "No")),
                internet_service=str(row.get("InternetService", "DSL")),
                online_security=str(row.get("OnlineSecurity", "No")),
                online_backup=str(row.get("OnlineBackup", "No")),
                device_protection=str(row.get("DeviceProtection", "No")),
                tech_support=str(row.get("TechSupport", "No")),
                streaming_tv=str(row.get("StreamingTV", "No")),
                streaming_movies=str(row.get("StreamingMovies", "No")),
                contract=str(row.get("Contract", "Month-to-month")),
                paperless_billing=str(row.get("PaperlessBilling", "Yes")),
                payment_method=str(row.get("PaymentMethod", "Electronic check")),
                monthly_charges=float(row.get("MonthlyCharges", 0.0)),
                total_charges=float(row.get("TotalCharges", 0.0)) if not pd.isna(row.get("TotalCharges")) else 0.0,
                churn=str(row.get("Churn", "No"))
            )
            db.add(cust_obj)
        db.commit()
        customers = db.query(Customer).all()
        
    # Convert SQLAlchemy items to DataFrame
    cust_dicts = []
    for c in customers:
        d = {col.name: getattr(c, col.name) for col in c.__table__.columns}
        d["customerID"] = d["customer_id"] # mapping
        cust_dicts.append(d)
        
    df_train = pd.DataFrame(cust_dicts)
    
    try:
        # Run ML training pipeline
        trained_pipelines, metrics = train_and_evaluate(df_train)
        
        # Save best model to disk
        best_name = save_best_pipeline(trained_pipelines, metrics)
        
        # Log to Audit table
        audit = AuditLog(
            action="RETRAIN_MODEL",
            details=f"Retrained models on {len(df_train)} rows. Best model: {best_name}. Metrics: {json.dumps(metrics[best_name])}"
        )
        db.add(audit)
        db.commit()
        
        # Re-run predictions on the database using the new best model
        logger.info("Updating predictions in database using the newly trained model...")
        db.query(Prediction).delete()
        db.commit()
        
        active_model = load_model()
        preprocessor = active_model["preprocessor"]
        model = active_model["model"]
        
        X_proc = preprocessor.transform(df_train)
        probs = model.predict_proba(X_proc)[:, 1]
        preds = model.predict(X_proc)
        
        for c, prob, pred in zip(customers, probs, preds):
            prob_val = float(prob)
            pred_val = int(pred)
            
            if prob_val >= 0.7:
                risk = "High"
            elif prob_val >= 0.3:
                risk = "Medium"
            else:
                risk = "Low"
                
            pred_obj = Prediction(
                customer_id=c.customer_id,
                churn_probability=prob_val,
                churn_prediction=pred_val,
                risk_category=risk,
                explainability=json.dumps({"factors": []}),
                model_version=best_name
            )
            db.add(pred_obj)
            
        db.commit()
        
        return {
            "status": "success",
            "best_model": best_name,
            "metrics": metrics[best_name],
            "all_metrics": metrics
        }
        
    except Exception as e:
        logger.error(f"Error training models: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to train models: {str(e)}")

@app.get("/api/explain/{customer_id}", response_model=ExplainResponse)
def explain_customer_churn(customer_id: str, db: Session = Depends(get_db)):
    """Computes SHAP values explaining the churn prediction for a specific customer."""
    if not is_model_trained():
        raise HTTPException(status_code=400, detail="No active model has been trained yet.")
        
    cust = db.query(Customer).filter(Customer.customer_id == customer_id).first()
    if not cust:
        raise HTTPException(status_code=404, detail=f"Customer with ID {customer_id} not found.")
        
    pred = db.query(Prediction).filter(Prediction.customer_id == customer_id).first()
    if not pred:
        raise HTTPException(status_code=400, detail=f"No prediction found for customer {customer_id}. Run training or predict first.")
        
    try:
        active_model = load_model()
        
        # Convert single customer to DataFrame row
        c_dict = {col.name: getattr(cust, col.name) for col in cust.__table__.columns}
        c_dict["customerID"] = c_dict["customer_id"]
        c_df = pd.DataFrame([c_dict])
        
        # Get SHAP explanations
        shap_factors = get_shap_explanation(c_df, active_model)
        
        formatted_factors = []
        for sf in shap_factors:
            formatted_factors.append({
                "feature": sf["feature"],
                "shap_value": sf["shap_value"],
                "importance": sf["importance"]
            })
            
        return {
            "customer_id": customer_id,
            "churn_probability": pred.churn_probability,
            "risk_category": pred.risk_category,
            "top_factors": formatted_factors
        }
        
    except Exception as e:
        logger.error(f"SHAP explanation failed for customer {customer_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to calculate SHAP: {str(e)}")

@app.post("/api/rag/query", response_model=RAGQueryResponse)
def query_rag_assistant(request: RAGQueryRequest, db: Session = Depends(get_db)):
    """Retrieves context from documentation and generates a conversational retention recommendation."""
    cust_context = None
    
    # If customer ID is provided, look up details to pass to RAG
    if request.customer_id:
        cust = db.query(Customer).filter(Customer.customer_id == request.customer_id).first()
        pred = db.query(Prediction).filter(Prediction.customer_id == request.customer_id).first()
        
        if cust and pred:
            cust_context = {
                "customer_id": cust.customer_id,
                "tenure": cust.tenure,
                "monthly_charges": cust.monthly_charges,
                "contract": cust.contract,
                "internet_service": cust.internet_service,
                "tech_support": cust.tech_support,
                "payment_method": cust.payment_method,
                "churn_probability": pred.churn_probability,
                "risk_category": pred.risk_category
            }
            
    try:
        # Run RAG chain
        rag_res = generate_retrieval_response(request.query, customer_context=cust_context)
        
        # Log to Audit
        audit = AuditLog(
            action="RAG_QUERY",
            details=f"Query: {request.query}. CustomerContext: {request.customer_id}. Mode: {rag_res['mode']}"
        )
        db.add(audit)
        db.commit()
        
        return rag_res
        
    except Exception as e:
        logger.error(f"RAG query execution failed: {e}")
        raise HTTPException(status_code=500, detail=f"RAG failure: {str(e)}")

@app.get("/api/analytics/summary", response_model=ExecutiveSummary)
def get_executive_summary(db: Session = Depends(get_db)):
    """Generates corporate KPIs and aggregated prediction metrics."""
    customers = db.query(Customer).all()
    predictions = db.query(Prediction).all()
    
    total_cust = len(customers)
    if total_cust == 0:
        return {
            "total_customers": 0,
            "churned_customers": 0,
            "predicted_churn_rate": 0.0,
            "actual_churn_rate": 0.0,
            "revenue_at_risk": 0.0,
            "average_tenure": 0.0,
            "risk_distribution": {"Low": 0, "Medium": 0, "High": 0}
        }
        
    # Actual churn rate (ground truth)
    actual_churn_count = sum(1 for c in customers if str(c.churn).strip().lower() == "yes")
    
    # Predicted statistics
    risk_counts = {"Low": 0, "Medium": 0, "High": 0}
    predicted_churn_count = 0
    revenue_at_risk = 0.0
    
    # Map predictions by customer_id
    pred_map = {p.customer_id: p for p in predictions}
    
    tenures = []
    for c in customers:
        tenures.append(c.tenure if c.tenure else 0)
        p = pred_map.get(c.customer_id)
        if p:
            risk_counts[p.risk_category] = risk_counts.get(p.risk_category, 0) + 1
            if p.churn_prediction == 1:
                predicted_churn_count += 1
                # Customer is predicted to leave, so their monthly spend is at risk
                revenue_at_risk += (c.monthly_charges if c.monthly_charges else 0.0)
                
    avg_tenure = float(np.mean(tenures)) if tenures else 0.0
    
    return {
        "total_customers": total_cust,
        "churned_customers": actual_churn_count,
        "predicted_churn_rate": float(predicted_churn_count / total_cust),
        "actual_churn_rate": float(actual_churn_count / total_cust),
        "revenue_at_risk": round(revenue_at_risk, 2),
        "average_tenure": round(avg_tenure, 2),
        "risk_distribution": risk_counts
    }

@app.get("/api/customers", response_model=List[Dict[str, Any]])
def list_customers(db: Session = Depends(get_db)):
    """Lists all customer profiles joined with their risk prediction details."""
    query = db.query(Customer, Prediction).outerjoin(
        Prediction, Customer.customer_id == Prediction.customer_id
    ).all()
    
    output = []
    for cust, pred in query:
        cust_dict = {col.name: getattr(cust, col.name) for col in cust.__table__.columns}
        
        if pred:
            cust_dict["churn_probability"] = pred.churn_probability
            cust_dict["churn_prediction"] = pred.churn_prediction
            cust_dict["risk_category"] = pred.risk_category
            cust_dict["model_version"] = pred.model_version
        else:
            cust_dict["churn_probability"] = None
            cust_dict["churn_prediction"] = None
            cust_dict["risk_category"] = "Unpredicted"
            cust_dict["model_version"] = "None"
            
        output.append(cust_dict)
        
    return output

@app.get("/api/audit-logs", response_model=List[Dict[str, Any]])
def get_audit_logs(db: Session = Depends(get_db)):
    """Returns the activity audit logs."""
    logs = db.query(AuditLog).order_by(AuditLog.timestamp.desc()).limit(50).all()
    return [
        {
            "id": l.id,
            "action": l.action,
            "details": l.details,
            "timestamp": l.timestamp.isoformat()
        }
        for l in logs
    ]
