import datetime
from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, DateTime, ForeignKey, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from src.config import DATABASE_URL

# Create database engine
engine = create_engine(
    DATABASE_URL, connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {}
)

# Session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base model class
Base = declarative_base()

class Customer(Base):
    __tablename__ = "customers"

    id = Column(Integer, primary_key=True, index=True)
    customer_id = Column(String, unique=True, index=True, nullable=False)
    name = Column(String, nullable=True)
    email = Column(String, nullable=True)
    gender = Column(String, nullable=True)
    senior_citizen = Column(Integer, nullable=True)  # 0 or 1
    partner = Column(String, nullable=True)          # Yes, No
    dependents = Column(String, nullable=True)       # Yes, No
    tenure = Column(Integer, nullable=True)          # Months
    phone_service = Column(String, nullable=True)    # Yes, No
    multiple_lines = Column(String, nullable=True)   # Yes, No, No phone service
    internet_service = Column(String, nullable=True) # DSL, Fiber optic, No
    online_security = Column(String, nullable=True)  # Yes, No, No internet service
    online_backup = Column(String, nullable=True)    # Yes, No, No internet service
    device_protection = Column(String, nullable=True) # Yes, No, No internet service
    tech_support = Column(String, nullable=True)     # Yes, No, No internet service
    streaming_tv = Column(String, nullable=True)     # Yes, No, No internet service
    streaming_movies = Column(String, nullable=True) # Yes, No, No internet service
    contract = Column(String, nullable=True)         # Month-to-month, One year, Two year
    paperless_billing = Column(String, nullable=True) # Yes, No
    payment_method = Column(String, nullable=True)    # Electronic check, Mailed check, Bank transfer, Credit card
    monthly_charges = Column(Float, nullable=True)
    total_charges = Column(Float, nullable=True)
    churn = Column(String, nullable=True)            # Yes, No, or None
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    # Relationships
    predictions = relationship("Prediction", back_populates="customer", cascade="all, delete-orphan")

class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    customer_id = Column(String, ForeignKey("customers.customer_id"), nullable=False)
    churn_probability = Column(Float, nullable=False)  # 0.0 to 1.0
    churn_prediction = Column(Integer, nullable=False)  # 0 or 1
    risk_category = Column(String, nullable=False)      # Low, Medium, High
    explainability = Column(Text, nullable=True)        # JSON string of contributing features
    model_version = Column(String, nullable=False)      # Model type (XGBoost, RandomForest, etc.)
    predicted_at = Column(DateTime, default=datetime.datetime.utcnow)

    # Relationships
    customer = relationship("Customer", back_populates="predictions")

class AuditLog(Base):
    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True, index=True)
    action = Column(String, nullable=False)             # UPLOAD_DATASET, RETRAIN_MODEL, RAG_QUERY, etc.
    details = Column(Text, nullable=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)

# Database helper functions
def init_db():
    Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
