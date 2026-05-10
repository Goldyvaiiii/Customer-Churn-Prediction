import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent.parent
SRC_DIR = BASE_DIR / "src"
DATA_DIR = BASE_DIR / "data"
DOCS_DIR = DATA_DIR / "docs"

# Create data directories if they don't exist
DATA_DIR.mkdir(exist_ok=True)
DOCS_DIR.mkdir(exist_ok=True)

# Configurations
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", 8000))
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{BASE_DIR}/database.sqlite")

# Vector Database
CHROMA_DB_DIR = os.getenv("CHROMA_DB_DIR", str(DATA_DIR / "chroma_db"))

# Machine Learning model storage
MODEL_SAVE_DIR = SRC_DIR / "ml" / "models"
MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)

# LLM Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL_NAME = os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
