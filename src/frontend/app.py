import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, List
import time
import json

# Import API helpers
import utils

# Page Configuration
st.set_page_config(
    page_title="Enterprise Churn Portal",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium Custom CSS styling (Inter font, dark glassmorphism cards, modern margins, custom badges)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Executive Card Styling */
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 24px;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(5px);
        -webkit-backdrop-filter: blur(5px);
        transition: transform 0.2s ease-in-out;
    }
    .metric-card:hover {
        transform: translateY(-4px);
        border-color: rgba(255, 255, 255, 0.25);
    }
    .metric-value {
        font-size: 32px;
        font-weight: 700;
        color: #FFFFFF;
        margin-top: 8px;
    }
    .metric-title {
        font-size: 14px;
        font-weight: 500;
        color: #8A99AD;
        text-transform: uppercase;
        letter-spacing: 0.8px;
    }
    .metric-change {
        font-size: 12px;
        font-weight: 600;
        margin-top: 4px;
    }
    .text-success { color: #10B981; }
    .text-danger { color: #EF4444; }
    .text-warning { color: #F59E0B; }
    
    /* Churn Badges */
    .badge {
        padding: 4px 8px;
        border-radius: 6px;
        font-size: 12px;
        font-weight: 600;
        text-transform: uppercase;
    }
    .badge-high { background-color: rgba(239, 68, 68, 0.15); color: #EF4444; border: 1px solid rgba(239, 68, 68, 0.3); }
    .badge-medium { background-color: rgba(245, 158, 11, 0.15); color: #F59E0B; border: 1px solid rgba(245, 158, 11, 0.3); }
    .badge-low { background-color: rgba(16, 185, 129, 0.15); color: #10B981; border: 1px solid rgba(16, 185, 129, 0.3); }
    
    /* Custom Sidebar Header */
    .sidebar-header {
        font-size: 20px;
        font-weight: 700;
        background: linear-gradient(135deg, #60A5FA 0%, #3B82F6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 24px;
    }
    
    /* Table styles */
    .stTable {
        border-radius: 8px;
        overflow: hidden;
    }
</style>
""", unsafe_allow_html=True)

# ----------------- SESSION STATE SETUP -----------------
if "active_page" not in st.session_state:
    st.session_state["active_page"] = "Dashboard"
if "selected_customer" not in st.session_state:
    st.session_state["selected_customer"] = None
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# ----------------- SIDEBAR NAVIGATION -----------------
with st.sidebar:
    st.markdown('<div class="sidebar-header">🔮 Churn Intelligence</div>', unsafe_allow_html=True)
    
    # Check API health and show status
    is_online = utils.check_backend_health()
    if is_online:
        st.sidebar.success("● API Service Connected")
    else:
        st.sidebar.info("⚡ Standalone Mode (No backend required)")
        
    st.markdown("---")
    
    # Navigation Buttons
    if st.sidebar.button("📊 Executive Dashboard", use_container_width=True):
        st.session_state["active_page"] = "Dashboard"
    if st.sidebar.button("👥 Customer Explorer", use_container_width=True):
        st.session_state["active_page"] = "Explorer"
    if st.sidebar.button("🧠 RAG Agent Assistant", use_container_width=True):
        st.session_state["active_page"] = "Assistant"
    if st.sidebar.button("⚙️ Model & System Manager", use_container_width=True):
        st.session_state["active_page"] = "Manager"
        
    st.markdown("---")
    st.markdown("<small>Enterprise Churn Platform v1.0<br>Developer Portfolio Mockup</small>", unsafe_allow_html=True)

# Show a dismissible notice if running in standalone mode (no backend)
if not is_online:
    st.info(
        "⚡ **Standalone Mode** — Running directly on Streamlit Cloud without a backend server. "
        "All ML, database, and RAG operations run inline. "
        "Head to **Model & System Manager → Seed Synthetic Dataset** to populate the dashboard.",
        icon="ℹ️"
    )

# ----------------- DASHBOARD PAGE -----------------
if st.session_state["active_page"] == "Dashboard":
    st.markdown("# 📊 Executive Retention Insights")
    st.markdown("Overview of key prediction metrics, revenue at risk, and risk cohort distributions.")
    
    # Fetch executive metrics
    try:
        summary = utils.get_executive_summary()
        
        # Display KPI cards
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Total Customers</div>
                <div class="metric-value">{summary['total_customers']:,}</div>
                <div class="metric-change text-success">Active Cohort</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            pred_rate = summary['predicted_churn_rate'] * 100
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Predicted Churn Rate</div>
                <div class="metric-value">{pred_rate:.1f}%</div>
                <div class="metric-change text-danger">ML Estimate</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Revenue At Risk</div>
                <div class="metric-value">${summary['revenue_at_risk']:,.2f}</div>
                <div class="metric-change text-warning">Monthly Loss Potential</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Avg Customer Tenure</div>
                <div class="metric-value">{summary['average_tenure']} mo</div>
                <div class="metric-change text-success">LTV Driver</div>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("###")
        
        # Charts section
        if summary['total_customers'] > 0:
            char_col1, char_col2 = st.columns([1, 2])
            
            with char_col1:
                st.markdown("##### Churn Risk Distribution")
                dist = summary['risk_distribution']
                fig_pie = px.pie(
                    names=list(dist.keys()),
                    values=list(dist.values()),
                    color=list(dist.keys()),
                    color_discrete_map={"Low": "#10B981", "Medium": "#F59E0B", "High": "#EF4444"},
                    hole=0.45
                )
                fig_pie.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font_color="#8A99AD",
                    legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
                )
                st.plotly_chart(fig_pie, use_container_width=True)
                
            with char_col2:
                st.markdown("##### Customer Churn Profile (Tenure vs Monthly Charges)")
                # Get customer list
                custs = utils.get_customers()
                if custs:
                    df = pd.DataFrame(custs)
                    
                    fig_scatter = px.scatter(
                        df,
                        x="tenure",
                        y="monthly_charges",
                        color="risk_category",
                        color_discrete_map={"Low": "#10B981", "Medium": "#F59E0B", "High": "#EF4444", "Unpredicted": "#8A99AD"},
                        labels={"tenure": "Tenure (Months)", "monthly_charges": "Monthly Charges ($)", "risk_category": "Risk Category"},
                        hover_data=["customer_id", "contract", "internet_service"]
                    )
                    fig_scatter.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font_color="#8A99AD",
                        xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)"),
                        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)")
                    )
                    st.plotly_chart(fig_scatter, use_container_width=True)
            
            # Additional analysis
            st.markdown("##### Churn Rate by Contract & Internet Service Types")
            col_bar1, col_bar2 = st.columns(2)
            
            df = pd.DataFrame(custs)
            with col_bar1:
                # Contract distribution
                contract_counts = df.groupby(["contract", "risk_category"]).size().reset_index(name="count")
                fig_contract = px.bar(
                    contract_counts,
                    x="contract",
                    y="count",
                    color="risk_category",
                    barmode="group",
                    color_discrete_map={"Low": "#10B981", "Medium": "#F59E0B", "High": "#EF4444", "Unpredicted": "#8A99AD"},
                    labels={"contract": "Contract Type", "count": "Customer Count", "risk_category": "Risk"}
                )
                fig_contract.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#8A99AD")
                st.plotly_chart(fig_contract, use_container_width=True)
                
            with col_bar2:
                # Internet service distribution
                internet_counts = df.groupby(["internet_service", "risk_category"]).size().reset_index(name="count")
                fig_internet = px.bar(
                    internet_counts,
                    x="internet_service",
                    y="count",
                    color="risk_category",
                    barmode="group",
                    color_discrete_map={"Low": "#10B981", "Medium": "#F59E0B", "High": "#EF4444", "Unpredicted": "#8A99AD"},
                    labels={"internet_service": "Internet Service Type", "count": "Customer Count", "risk_category": "Risk"}
                )
                fig_internet.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#8A99AD")
                st.plotly_chart(fig_internet, use_container_width=True)
                
        else:
            st.info("No customer data loaded. Please upload a dataset or seed the database in the Model & System Manager section.")
            
    except Exception as e:
        st.error(f"Error rendering dashboard: {e}")

# ----------------- CUSTOMER EXPLORER PAGE -----------------
elif st.session_state["active_page"] == "Explorer":
    st.markdown("# 👥 Customer Risk Explorer")
    st.markdown("Search individual accounts, inspect their risk factors, and construct personalized retention plays.")
    
    try:
        custs = utils.get_customers()
        if not custs:
            st.info("No customers found. Please train a model or seed data first in the Manager tab.")
            st.stop()
            
        df = pd.DataFrame(custs)
        
        # Filtering controls
        search_col, risk_col = st.columns([2, 1])
        with search_col:
            search_query = st.text_input("🔍 Search by Customer ID:", "")
        with risk_col:
            risk_filter = st.selectbox("Filter by Churn Risk:", ["All", "High", "Medium", "Low"])
            
        # Apply filters
        filtered_df = df.copy()
        if search_query:
            filtered_df = filtered_df[filtered_df["customer_id"].str.contains(search_query, case=False)]
        if risk_filter != "All":
            filtered_df = filtered_df[filtered_df["risk_category"] == risk_filter]
            
        # Grid layout: left column list, right column detailed profile
        list_col, detail_col = st.columns([1, 1.2])
        
        with list_col:
            st.markdown(f"**Matching Accounts ({len(filtered_df)})**")
            
            # Select Customer from Table
            # Create readable table
            display_df = filtered_df[["customer_id", "risk_category", "churn_probability", "contract", "monthly_charges"]].copy()
            if "churn_probability" in display_df.columns:
                display_df["churn_probability"] = display_df["churn_probability"].apply(
                    lambda x: f"{x*100:.1f}%" if pd.notna(x) else "N/A"
                )
                
            # Create a selection radio or selectbox to inspect details
            cust_ids = list(filtered_df["customer_id"].values)
            if cust_ids:
                selected_id = st.selectbox(
                    "Select Customer ID to inspect:",
                    cust_ids,
                    index=0 if st.session_state["selected_customer"] not in cust_ids else cust_ids.index(st.session_state["selected_customer"])
                )
                st.session_state["selected_customer"] = selected_id
                
                # Render styled colorful table
                def style_risk(row):
                    risk = row["risk_category"]
                    if risk == "High":
                        return ["background-color: rgba(239, 68, 68, 0.15); color: #EF4444; font-weight: 500;"] * len(row)
                    elif risk == "Medium":
                        return ["background-color: rgba(245, 158, 11, 0.15); color: #F59E0B; font-weight: 500;"] * len(row)
                    elif risk == "Low":
                        return ["background-color: rgba(16, 185, 129, 0.15); color: #10B981; font-weight: 500;"] * len(row)
                    else:
                        return [""] * len(row)

                indexed_df = display_df.set_index("customer_id")
                styled_df = indexed_df.style.apply(style_risk, axis=1)
                st.dataframe(styled_df, use_container_width=True)
            else:
                st.warning("No customers match the active filters.")
                st.session_state["selected_customer"] = None
                
        with detail_col:
            if st.session_state["selected_customer"]:
                selected_cust = df[df["customer_id"] == st.session_state["selected_customer"]].iloc[0]
                
                st.markdown(f"### 👤 Profile: `{selected_cust['customer_id']}`")
                
                # Show risk tier badge
                risk = selected_cust['risk_category']
                prob = selected_cust['churn_probability']
                badge_class = "badge-low" if risk == "Low" else "badge-medium" if risk == "Medium" else "badge-high"
                prob_text = f"{prob*100:.1f}% Churn Probability" if pd.notna(prob) else "Uncalculated"
                
                st.markdown(f"""
                <span class="badge {badge_class}">{risk} Risk</span> &nbsp;&nbsp;&nbsp; <b>{prob_text}</b>
                """, unsafe_allow_html=True)
                st.markdown("###")
                
                # Demographic and Billing details
                det_tab1, det_tab2, det_tab3 = st.tabs(["Billing & Services", "XAI (SHAP Factors)", "Retention Plan Guide"])
                
                with det_tab1:
                    col_p1, col_p2 = st.columns(2)
                    with col_p1:
                        st.markdown("**Demographics**")
                        st.markdown(f"- **Gender**: {selected_cust.get('gender')}")
                        st.markdown(f"- **Senior Citizen**: {'Yes' if selected_cust.get('senior_citizen') == 1 else 'No'}")
                        st.markdown(f"- **Partner**: {selected_cust.get('partner')}")
                        st.markdown(f"- **Dependents**: {selected_cust.get('dependents')}")
                        st.markdown(f"- **Tenure**: {selected_cust.get('tenure')} months")
                        st.markdown(f"- **Contract**: {selected_cust.get('contract')}")
                        
                    with col_p2:
                        st.markdown("**Services**")
                        st.markdown(f"- **Internet Service**: {selected_cust.get('internet_service')}")
                        st.markdown(f"- **Tech Support**: {selected_cust.get('tech_support')}")
                        st.markdown(f"- **Online Security**: {selected_cust.get('online_security')}")
                        st.markdown(f"- **Online Backup**: {selected_cust.get('online_backup')}")
                        st.markdown(f"- **Payment Method**: {selected_cust.get('payment_method')}")
                        st.markdown(f"- **Monthly Charges**: ${selected_cust.get('monthly_charges'):.2f}")
                        st.markdown(f"- **Total Charges**: ${selected_cust.get('total_charges'):.2f}")
                        
                with det_tab2:
                    st.markdown("**SHAP Explainability Insights**")
                    st.markdown("These are the top factors contributing to this customer's risk prediction:")
                    
                    if pd.notna(prob):
                        # Fetch SHAP factors from API
                        with st.spinner("Calculating SHAP forces..."):
                            try:
                                shap_res = utils.get_shap_explanation(selected_cust['customer_id'])
                                factors = shap_res["top_factors"]
                                
                                if factors:
                                    # Plot interactive horizontal bar chart
                                    f_df = pd.DataFrame(factors)
                                    # Format feature names nicely
                                    f_df["feature"] = f_df["feature"].str.replace("_", " ")
                                    
                                    # Positive SHAP pushes churn likelihood up, negative down
                                    f_df["color"] = f_df["shap_value"].apply(lambda val: "#EF4444" if val > 0 else "#10B981")
                                    
                                    fig_shap = go.Figure()
                                    fig_shap.add_trace(go.Bar(
                                        y=f_df["feature"],
                                        x=f_df["shap_value"],
                                        orientation="h",
                                        marker_color=f_df["color"],
                                        text=f_df["shap_value"].apply(lambda v: f"+{v:.3f}" if v > 0 else f"{v:.3f}"),
                                        textposition="inside"
                                    ))
                                    fig_shap.update_layout(
                                        title=dict(text="SHAP Feature Forces (Red = Increases Churn Risk, Green = Decreases Churn Risk)", font_size=12),
                                        paper_bgcolor="rgba(0,0,0,0)",
                                        plot_bgcolor="rgba(0,0,0,0)",
                                        font_color="#8A99AD",
                                        yaxis=dict(autorange="reversed"),
                                        margin=dict(l=10, r=10, t=30, b=10)
                                    )
                                    st.plotly_chart(fig_shap, use_container_width=True)
                                else:
                                    st.info("No explainability features returned.")
                            except Exception as e:
                                st.error(f"Failed to load SHAP explainability: {e}")
                    else:
                        st.warning("Train the model first to generate predictions and explainability logs.")
                        
                with det_tab3:
                    st.markdown("**RAG Assistant Integration**")
                    st.markdown("Query the knowledge base using this customer's context to get specific action guides:")
                    
                    # Custom button to generate strategy
                    if st.button("🧙 Generate Retention Playbook", use_container_width=True):
                        with st.spinner("Consulting vector guides..."):
                            try:
                                prompt = f"What is the best customer retention strategy for customer {selected_cust['customer_id']}?"
                                response = utils.ask_rag_assistant(prompt, customer_id=selected_cust['customer_id'])
                                
                                st.markdown(response["response"])
                                
                                # Show Sources used
                                st.markdown("##### Sources Utilized:")
                                for s in response["sources"]:
                                    st.markdown(f"- `{s.get('source')}` (chunk #{s.get('chunk_index')})")
                            except Exception as e:
                                st.error(f"RAG playbooks error: {e}")
                                
            else:
                st.info("Select a customer from the table or dropdown on the left to review detail panes.")
                
    except Exception as e:
        st.error(f"Error rendering customer explorer: {e}")

# ----------------- RAG ASSISTANT PAGE -----------------
elif st.session_state["active_page"] == "Assistant":
    st.markdown("# 🧠 RAG Churn Assistant")
    st.markdown("Retrieve retention tactics, SaaS/telecom onboarding playbooks, and contract optimization guides.")
    
    # Initialize message list
    for msg in st.session_state["chat_history"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
    # Quick queries
    st.markdown("💡 **Sample Retention Questions:**")
    q_col1, q_col2, q_col3 = st.columns(3)
    with q_col1:
        if st.button("How to reduce month-to-month fiber optic churn?"):
            st.session_state["chat_input"] = "How can we reduce churn for month-to-month fiber optic customers?"
    with q_col2:
        if st.button("What is a customer health score matrix?"):
            st.session_state["chat_input"] = "Explain the customer health scoring matrix and weights."
    with q_col3:
        if st.button("Best workflows for high charges/low support?"):
            st.session_state["chat_input"] = "What retention workflow should be used for clients paying premium monthly charges who rarely use support?"
            
    # Formulate inputs
    user_query = st.chat_input("Ask a question about churn mitigation or retention playbooks:")
    
    # Handle quick click
    if "chat_input" in st.session_state and st.session_state["chat_input"]:
        user_query = st.session_state["chat_input"]
        st.session_state["chat_input"] = "" # Reset
        
    if user_query:
        # Display user query
        with st.chat_message("user"):
            st.markdown(user_query)
        st.session_state["chat_history"].append({"role": "user", "content": user_query})
        
        with st.spinner("Retrieving document chunks and generating advice..."):
            try:
                # Call RAG Assistant
                response = utils.ask_rag_assistant(user_query)
                
                ans = response["response"]
                
                # Append sources details
                ans += "\n\n**Sources referenced:**\n"
                for idx, src in enumerate(response["sources"]):
                    ans += f"- `{src.get('source')}` (chunk #{src.get('chunk_index')})\n"
                    
                # Display response
                with st.chat_message("assistant"):
                    st.markdown(ans)
                    
                st.session_state["chat_history"].append({"role": "assistant", "content": ans})
            except Exception as e:
                st.error(f"Error querying RAG assistant: {e}")

# ----------------- MANAGER PAGE -----------------
elif st.session_state["active_page"] == "Manager":
    st.markdown("# ⚙️ Model & System Manager")
    st.markdown("Upload corporate customer records, retrain predictions models, inspect model configurations, and view audits logs.")
    
    tab_m1, tab_m2, tab_m3 = st.tabs(["Data & Model Ingestion", "Model Performance & Metrics", "System Audit Logs"])
    
    with tab_m1:
        st.markdown("### 📥 Import Customer Dataset")
        st.markdown("Upload customer profiles in CSV format. The CSV must contain standard columns (like gender, tenure, contract, MonthlyCharges, etc.).")
        
        uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
        if uploaded_file is not None:
            if st.button("Import and Predict Churn Rates", use_container_width=True):
                with st.spinner("Importing dataset, clearing old databases, and auto-running churn predictions..."):
                    try:
                        file_bytes = uploaded_file.read()
                        res = utils.upload_csv(file_bytes, uploaded_file.name)
                        st.success(f"Success! Imported {res['records_count']} rows. Generated {res['predictions_generated']} probability predictions.")
                    except Exception as e:
                        st.error(f"File upload failed: {e}")
                        
        st.markdown("---")
        st.markdown("### 🧬 Need dummy data?")
        st.markdown("If you don't have a dataset ready, seed our SQLite database with 1,000 synthetic records with realistic churn biases.")
        
        if st.button("Seed Synthetic Dataset & Train All Models", use_container_width=True):
            with st.spinner("Seeding database and running training pipeline..."):
                try:
                    res = utils.trigger_training()
                    st.success(f"Success! Seeded synthetic dataset and trained active model: {res['best_model']}")
                    st.info(f"Model Metrics: Accuracy: {res['metrics']['accuracy']:.3f} | F1: {res['metrics']['f1_score']:.3f} | AUC: {res['metrics']['roc_auc']:.3f}")
                except Exception as e:
                    st.error(f"Training failed: {e}")
                    
    with tab_m2:
        st.markdown("### 📊 Model Selection & Performance")
        
        # Check active model status
        try:
            # We will try to read customers to see if model details are available
            custs = utils.get_customers()
            if custs:
                df = pd.DataFrame(custs)
                model_name = df["model_version"].iloc[0] if "model_version" in df.columns else "None"
                
                st.info(f"Active Prediction Model: **{model_name}**")
                
                # Fetch recent model stats from audit logs or display metrics table
                logs = utils.get_audit_logs()
                training_log = next((l for l in logs if l["action"] == "RETRAIN_MODEL"), None)
                
                if training_log:
                    st.markdown("#### Model Training Metrics (Comparison)")
                    # Parse metrics JSON
                    details = training_log["details"]
                    # Extract from log details string
                    # Best model: XGBoost. Metrics: {"LogisticRegression": {...}, ...}
                    try:
                        metrics_part = details.split("Metrics: ")[1]
                        metrics_dict = json.loads(metrics_part)
                        
                        m_df = pd.DataFrame(metrics_dict).T
                        st.table(m_df)
                        
                        # Plotly metrics comparison
                        m_df_melt = m_df.reset_index().melt(id_vars="index", var_name="Metric", value_name="Score")
                        m_df_melt.columns = ["Model", "Metric", "Score"]
                        
                        fig_metrics = px.bar(
                            m_df_melt,
                            x="Metric",
                            y="Score",
                            color="Model",
                            barmode="group",
                            title="Classifier Performance Metrics Comparison"
                        )
                        fig_metrics.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#8A99AD")
                        st.plotly_chart(fig_metrics, use_container_width=True)
                        
                    except Exception as ex:
                        st.warning("Detailed metrics parse error, showing raw log.")
                        st.code(training_log["details"])
                else:
                    st.warning("No training log found in system audits. Trigger a model training run first.")
            else:
                st.warning("No customers or trained models exist in the database.")
        except Exception as e:
            st.error(f"Error fetching model performance: {e}")
            
    with tab_m3:
        st.markdown("### 🗂️ Application Activity Audit Logs")
        st.markdown("Audit logs recording major system events, dataset uploads, training sessions, and user questions.")
        
        try:
            logs = utils.get_audit_logs()
            if logs:
                log_df = pd.DataFrame(logs)
                st.dataframe(log_df[["timestamp", "action", "details"]].set_index("timestamp"), use_container_width=True)
            else:
                st.info("Audit logs table is currently empty.")
        except Exception as e:
            st.error(f"Failed to fetch audit logs: {e}")
