import os
import pickle
import logging
from typing import Dict, Any, Optional
from src.config import MODEL_SAVE_DIR

logger = logging.getLogger(__name__)

def get_model_path() -> str:
    """Returns the absolute path to the active model pickle file."""
    return os.path.join(MODEL_SAVE_DIR, "active_model.pkl")

def is_model_trained() -> bool:
    """Checks if a trained model is available on disk."""
    return os.path.exists(get_model_path())

def load_model() -> Optional[Dict[str, Any]]:
    """Loads and returns the active model pipeline (model, preprocessor, features, metrics)."""
    if not is_model_trained():
        return None
    try:
        with open(get_model_path(), "rb") as f:
            return pickle.load(f)
    except Exception as e:
        logger.error(f"Error loading active model: {e}")
        return None
