# Enterprise AI Customer Churn Prediction & Retention Platform

An end-to-end, production-grade application that combines **Supervised Machine Learning Classification**, **Explainable AI (SHAP)**, and **Retrieval-Augmented Generation (RAG)** to predict customer churn risks and generate personalized, playbook-driven retention strategies for customer success teams.

---

## 🚀 Key Features

1. **Multi-Model Machine Learning Pipeline**
   - Automatically preprocesses, handles missing values, scales, and encodes customer tabular records.
   - Trains and evaluates three classifiers: **Logistic Regression**, **Random Forest**, and **XGBoost**.
   - Compares performance metrics (Accuracy, Precision, Recall, F1-Score, ROC-AUC) and automatically deploys the best performing model.
   - Supports uploading custom customer CSV datasets.

2. **Explainable AI (XAI)**
   - Computes local **SHAP (SHapley Additive exPlanations)** values in real-time for any selected customer.
   - Visualizes positive and negative feature forces driving a customer's churn risk probability, avoiding "black box" machine learning predictions.

3. **RAG-Powered Retention Assistant**
   - Ingests internal retention playbooks, SaaS customer success manuals, and telecom service guides.
   - Chunks and stores documents inside a persistent **ChromaDB** vector database.
   - Dual-mode retrieval: Utilizes OpenAI (`gpt-4o-mini`) when configured, and falls back to a rules-based template synthesizer in offline mode.
   - Dynamically pulls customer-specific risk profile metrics as context to generate personalized mitigation advice.

4. **Executive Dashboard UI**
   - Premium, responsive Streamlit dashboard with HSL coordinated dark-mode aesthetics.
   - Real-time KPI counters: Total Accounts, Average Tenure, Monthly Revenue at Risk, and Predicted Churn Rates.
   - Interactive Plotly visualizations for churn profile distributions and feature metrics.
   - Searchable customer explorer tables.

5. **FastAPI Backend & Database**
   - Clean, modular folder structure separating API, ML, and RAG layers.
   - Persistent SQLAlchemy database layer (SQLite) tracking customers, logs, and prediction histories.

---

## 🛠️ Tech Stack

*   **Language**: Python 3.12+
*   **Machine Learning**: Scikit-Learn, XGBoost, SHAP, Pandas, NumPy
*   **Vector Search & RAG**: ChromaDB (ONNX Embeddings), LangChain, OpenAI
*   **Backend API**: FastAPI, Uvicorn, SQLAlchemy
*   **Frontend Dashboard**: Streamlit, Plotly, HTML5/CSS3
*   **Deployment**: Docker, Docker Compose

---

## 📁 Repository Structure

```text
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
├── .gitignore
├── README.md
├── database.sqlite            # SQLAlchemy SQLite database (created on start)
├── data/
│   ├── docs/                  # Raw retention playbooks (vector db context)
│   │   ├── retention_strategies.md
│   │   └── saas_churn_guide.md
│   └── chroma_db/             # Persistent ChromaDB vector files
├── src/
│   ├── config.py              # Configuration & env management
│   ├── database.py            # SQLite schema models and session setup
│   ├── ml/
│   │   ├── pipeline.py        # Preprocessing, ML classifiers, and SHAP
│   │   └── models.py          # Active model pickle loader
│   ├── rag/
│   │   ├── vectorstore.py     # Document chunks indexer and vector store
│   │   └── retriever.py       # Query parser and LLM connector
│   ├── api/
│   │   ├── main.py            # FastAPI main app and endpoints
│   │   └── schemas.py         # Request and response Pydantic models
│   └── frontend/
│       ├── app.py             # Streamlit graphical UI
│       └── utils.py           # Streamlit API callers
```

---

## ⚙️ Installation & Quick Start

### Option A: Local Deployment (Python Virtual Environment)

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Goldyvaiiii/Customer-Churn-Prediction.git
   cd Customer-Churn-Prediction
   ```

2. **Create and Activate a Virtual Environment**:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Configure Environment Variables**:
   Copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```
   *Note: The platform is designed to run out-of-the-box in local offline mode. However, if you wish to use OpenAI for generative conversational chat, fill in your `OPENAI_API_KEY` in the `.env` file.*

5. **Start the FastAPI Backend Service**:
   ```bash
   uvicorn src.api.main:app --host 127.0.0.1 --port 8000 --reload
   ```

6. **Start the Streamlit Frontend Service** (In a new terminal window):
   ```bash
   streamlit run src/frontend/app.py
   ```
   *The browser will open automatically at `http://localhost:8501`.*

---

### Option B: Docker Deployment (Single Command)

Ensure you have Docker and Docker Compose installed, then run:

```bash
docker-compose up --build
```

This single command:
1. Compiles the Dockerfile.
2. Starts the FastAPI server on port `8000`.
3. Starts the Streamlit dashboard on port `8501`.
4. Binds local folders for persistent DB and models caching.

Access the services:
- **FastAPI Documentation (Swagger)**: `http://localhost:8000/docs`
- **Streamlit Web Dashboard**: `http://localhost:8501`

---

## 🔌 API Reference Summary

| Method | Endpoint | Description |
|:---|:---|:---|
| **POST** | `/api/upload` | Upload a custom customer CSV dataset (populates database). |
| **POST** | `/api/train` | Train and deploy best model (LogisticRegression, RandomForest, XGBoost). |
| **GET** | `/api/customers` | Retrieve all customer profiles joined with predictions. |
| **GET** | `/api/explain/{customer_id}` | Calculate and return SHAP force values for a customer. |
| **POST** | `/api/rag/query` | Submit query to the RAG retention assistant. |
| **GET** | `/api/analytics/summary` | Retrieve high-level corporate KPIs and risk distribution. |
| **GET** | `/api/audit-logs` | Retrieve recent application activity log. |

---

## 📈 Portfolio Resume Impact

Developing this end-to-end platform demonstrates several core backend and machine learning engineering competencies:
*   **Full-Stack AI Application Design**: Connects modular APIs (FastAPI) to interactive frontends (Streamlit) using standard HTTP schemas.
*   **Explainable Machine Learning**: Uses Shapley values to make AI predictions interpretable, a critical requirement for enterprise financial risk mitigation.
*   **Retrieval-Augmented Generation**: Combines vector databases (ChromaDB) and local ONNX embedding models to ground large language models with company-specific knowledge.
*   **Production Deployment Readiness**: Uses Docker containerization and environment variables, preparing the platform for modern cloud hosting environments like AWS, GCP, or Render.
