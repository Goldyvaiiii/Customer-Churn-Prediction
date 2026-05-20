import requests
import os
import pandas as pd
import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

# FastAPI base URL (can be customized via environment variable)
API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")

def check_backend_health() -> bool:
    """Checks if the FastAPI backend is running and reachable."""
    try:
        response = requests.get(f"{API_BASE_URL}/", timeout=2)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False

def upload_csv(file_bytes: bytes, filename: str) -> Dict[str, Any]:
    """Uploads a CSV file to the backend."""
    files = {"file": (filename, file_bytes, "text/csv")}
    response = requests.post(f"{API_BASE_URL}/api/upload", files=files)
    response.raise_for_status()
    return response.json()

def trigger_training() -> Dict[str, Any]:
    """Triggers the ML model training pipeline."""
    response = requests.post(f"{API_BASE_URL}/api/train")
    response.raise_for_status()
    return response.json()

def get_customers() -> List[Dict[str, Any]]:
    """Fetches all customer records and predictions."""
    response = requests.get(f"{API_BASE_URL}/api/customers")
    response.raise_for_status()
    return response.json()

def get_executive_summary() -> Dict[str, Any]:
    """Fetches the executive KPI analytics summary."""
    response = requests.get(f"{API_BASE_URL}/api/analytics/summary")
    response.raise_for_status()
    return response.json()

def get_shap_explanation(customer_id: str) -> Dict[str, Any]:
    """Fetches the SHAP explanation factors for a customer."""
    response = requests.get(f"{API_BASE_URL}/api/explain/{customer_id}")
    response.raise_for_status()
    return response.json()

def ask_rag_assistant(query: str, customer_id: Optional[str] = None) -> Dict[str, Any]:
    """Sends a query to the RAG retention assistant."""
    payload = {"query": query, "customer_id": customer_id}
    response = requests.post(f"{API_BASE_URL}/api/rag/query", json=payload)
    response.raise_for_status()
    return response.json()

def get_audit_logs() -> List[Dict[str, Any]]:
    """Fetches the application activity audit logs."""
    response = requests.get(f"{API_BASE_URL}/api/audit-logs")
    response.raise_for_status()
    return response.json()
