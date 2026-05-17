from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime

class CustomerBase(BaseModel):
    customer_id: str = Field(..., alias="customerID")
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: Optional[float] = None
    Churn: Optional[str] = None

    class Config:
        populate_by_name = True
        from_attributes = True

class CustomerResponse(CustomerBase):
    id: int
    created_at: datetime

class PredictionResponse(BaseModel):
    customer_id: str
    churn_probability: float
    churn_prediction: int
    risk_category: str
    model_version: str
    predicted_at: datetime

class CustomerPredictRequest(BaseModel):
    customer_id: str

class RAGQueryRequest(BaseModel):
    query: str
    customer_id: Optional[str] = None

class RAGQueryResponse(BaseModel):
    query: str
    response: str
    mode: str
    sources: List[Dict[str, Any]]

class TrainingSummary(BaseModel):
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: float

class TrainResponse(BaseModel):
    status: str
    best_model: str
    metrics: TrainingSummary
    all_metrics: Dict[str, TrainingSummary]

class SHAPFactor(BaseModel):
    feature: str
    shap_value: float
    importance: float

class ExplainResponse(BaseModel):
    customer_id: str
    churn_probability: float
    risk_category: str
    top_factors: List[SHAPFactor]

class ExecutiveSummary(BaseModel):
    total_customers: int
    churned_customers: int
    predicted_churn_rate: float
    actual_churn_rate: float
    revenue_at_risk: float
    average_tenure: float
    risk_distribution: Dict[str, int]
