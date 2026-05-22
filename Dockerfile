FROM python:3.12-slim

# Prevent Python from writing pyc files to disk and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Install system dependencies needed for compiling ML packages (e.g. SHAP, XGBoost)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy and install python dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY src/ /app/src/
COPY data/ /app/data/
COPY .env.example /app/.env

# Create entrypoint shell script
RUN echo '#!/bin/bash\n\
echo "Starting FastAPI Backend..."\n\
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 &\n\
\n\
echo "Starting Streamlit Frontend..."\n\
streamlit run src/frontend/app.py --server.port 8501 --server.address 0.0.0.0\n\
' > /app/entrypoint.sh && chmod +x /app/entrypoint.sh

# Expose backend (8000) and frontend (8501) ports
EXPOSE 8000
EXPOSE 8501

ENTRYPOINT ["/app/entrypoint.sh"]
