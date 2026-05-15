import logging
from typing import Dict, Any, List, Optional
import requests
import json

from src.config import OPENAI_API_KEY, OPENAI_MODEL_NAME, OPENAI_API_BASE
from src.rag.vectorstore import query_vectorstore

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def generate_retrieval_response(
    query: str, 
    customer_context: Optional[Dict[str, Any]] = None,
    n_results: int = 3
) -> Dict[str, Any]:
    """
    Retrieves relevant playbooks and generates a retention recommendation.
    If OpenAI is not configured, it uses a high-quality local template synthesizer.
    """
    logger.info(f"RAG Query received: '{query}'")
    
    # Step 1: Query vector store
    retrieved_docs = query_vectorstore(query, n_results=n_results)
    
    # Combine retrieved contexts
    context_text = "\n\n".join([
        f"[Source: {doc['metadata'].get('source', 'Unknown')}]\n{doc['content']}"
        for doc in retrieved_docs
    ])
    
    # Prepare customer profile summary if context is provided
    customer_summary = ""
    if customer_context:
        customer_summary = (
            f"Customer Profile:\n"
            f"- ID: {customer_context.get('customer_id')}\n"
            f"- Tenure: {customer_context.get('tenure')} months\n"
            f"- Monthly Charges: ${customer_context.get('monthly_charges')}\n"
            f"- Contract: {customer_context.get('contract')}\n"
            f"- Internet Service: {customer_context.get('internet_service')}\n"
            f"- Tech Support: {customer_context.get('tech_support')}\n"
            f"- Payment Method: {customer_context.get('payment_method')}\n"
            f"- Predicted Churn Probability: {customer_context.get('churn_probability', 0.0) * 100:.1f}%\n"
            f"- Risk Tier: {customer_context.get('risk_category')}\n"
        )
        
    # Check if OpenAI API Key is valid
    is_openai_configured = (
        OPENAI_API_KEY 
        and OPENAI_API_KEY != "your_openai_api_key_here" 
        and len(OPENAI_API_KEY.strip()) > 10
    )
    
    if is_openai_configured:
        try:
            logger.info("Using OpenAI for response generation...")
            response_text = _call_openai_llm(query, context_text, customer_summary)
            mode = "OpenAI-RAG"
        except Exception as e:
            logger.error(f"OpenAI RAG failed, falling back to local synthesis: {e}")
            response_text = _local_synthesizer(query, retrieved_docs, customer_context)
            mode = "Local-Synthesis (Fallback)"
    else:
        logger.info("OpenAI API key not configured. Using local template synthesis...")
        response_text = _local_synthesizer(query, retrieved_docs, customer_context)
        mode = "Local-Synthesis (Offline Mode)"
        
    return {
        "query": query,
        "response": response_text,
        "mode": mode,
        "sources": [doc["metadata"] for doc in retrieved_docs]
    }

def _call_openai_llm(query: str, context: str, customer_summary: str) -> str:
    """Calls the OpenAI-compatible API to generate a response."""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY}"
    }
    
    system_prompt = (
        "You are an enterprise Customer Retention Assistant. Your job is to help Customer Success Managers "
        "and account executives prevent customer churn by providing structured, highly actionable advice. "
        "Use the retrieved retention playbook and customer details below to answer the user's question. "
        "Keep recommendations specific, practical, and business-focused. Include step-by-step instructions. "
        "Always cite which playbook rules you are utilizing."
    )
    
    user_content = f"User Question: {query}\n\n"
    if customer_summary:
        user_content += f"{customer_summary}\n\n"
    user_content += f"Retrieved Context:\n{context}\n"
    
    payload = {
        "model": OPENAI_MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ],
        "temperature": 0.3
    }
    
    url = f"{OPENAI_API_BASE}/chat/completions"
    response = requests.post(url, headers=headers, json=payload, timeout=15)
    response.raise_for_status()
    
    result = response.json()
    return result["choices"][0]["message"]["content"]

def _local_synthesizer(query: str, docs: List[Dict[str, Any]], customer: Optional[Dict[str, Any]]) -> str:
    """Synthesizes playbooks and customer info into a neat markdown response without external APIs."""
    if not docs:
        return (
            "### 📭 No Information Found\n"
            "I couldn't find any relevant retention strategies in the knowledge base. "
            "Please upload or create markdown playbook files in the `data/docs/` folder."
        )
        
    main_doc = docs[0]["content"]
    source_file = docs[0]["metadata"].get("source", "playbook")
    
    # Build a clean response structure
    res = []
    res.append(f"### 📋 Strategic Action Plan (Offline Knowledge Retrieval)")
    res.append(f"*Source Document: `{source_file}`*")
    res.append("")
    
    if customer:
        res.append(f"#### 🎯 Summary for Customer `{customer.get('customer_id')}`")
        res.append(f"- **Risk Category**: `{customer.get('risk_category')}` (Churn Prob: **{customer.get('churn_probability', 0.0) * 100:.1f}%**)")
        res.append(f"- **Contract**: {customer.get('contract')} | **Charges**: ${customer.get('monthly_charges')}/mo")
        res.append("")
        
        # Craft direct advice based on risk
        risk = customer.get("risk_category", "").upper()
        if "HIGH" in risk:
            res.append(
                "⚠️ **Immediate High-Risk Actions Required:**\n"
                "1. **Schedule Call**: CSM to coordinate an emergency review call with the customer within 24 hours.\n"
                "2. **Financial Incentive**: Offer a renewal incentive (e.g., 20% discount on a 1-year contract conversion).\n"
                "3. **Friction Review**: Audit support logs to address unresolved technical/billing complaints."
            )
        elif "MEDIUM" in risk:
            res.append(
                "⚡ **Proactive Medium-Risk Actions Required:**\n"
                "1. **Onboarding Check**: Send a tailored training session invitation to boost product engagement.\n"
                "2. **Loyalty Promotion**: Offer a free premium add-on upgrade or loyalty billing credits for 3 months.\n"
                "3. **Plan Migration**: Pitch a transition from Month-to-Month to a stable 1-year discounted rate."
            )
        else:
            res.append(
                "🟢 **Low-Risk Account Maintenance:**\n"
                "1. **Survey**: Send an Net Promoter Score (NPS) survey to evaluate satisfaction.\n"
                "2. **Upsell**: Identify cross-selling opportunities for supplementary premium features."
            )
        res.append("")
        
    res.append("#### 📖 Relevant Knowledge Base Insights")
    res.append("Here is the most relevant snippet from our playbooks:")
    res.append("")
    
    # Expose the most relevant chunk
    cleaned_chunk = main_doc.strip()
    # Indent it or format as markdown block
    res.append(f"> {cleaned_chunk}")
    res.append("")
    res.append("---")
    res.append("ℹ️ *Note: This response was generated locally using semantic search over internal documents. "
               "To enable full generative conversational responses, configure your `OPENAI_API_KEY` in the settings page.*")
    
    return "\n".join(res)
