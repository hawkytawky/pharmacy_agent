"""Documentation page for the Pharmacy Assistant."""

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))
from auth import login_barrier
from components import hide_default_nav, render_sidebar

login_barrier()
st.set_page_config(page_title="Documentation", page_icon="📄", layout="centered")
hide_default_nav()
render_sidebar()

st.title("📄 Documentation")
st.caption("System Architecture, Tech Stack, and Limitations.")

# --- 1. SYSTEM DESIGN ---
st.header("1. System Design")

st.markdown("### 🏗 Architecture: The ReAct Agent")
st.write("""
The core of the application is built on a **ReAct (Reasoning + Acting)** architecture.
This design allows the LLM to pause, "think" about the user's request, select the appropriate
tool to retrieve missing information, and then "act" by synthesizing the final answer.
""")

# --- 2. RAG & SEARCH (NEW SECTION) ---
st.markdown("### 🔍 Semantic Search")
st.write("""
To ensure robust entity resolution, the system avoids brittle string matching
and employs a **RAG (Retrieval-Augmented Generation)** approach.
""")

st.write("""
**Example:** A broad user query like *"Paracetamol"* correctly resolves to the specific database entry *"Paracetamol-ratiopharm 500mg"*, whereas exact string matching would fail.
""")

st.markdown("### 🛠 The Tool Suite")
st.write(
    "The agent interacts with a Supabase backend via four distinct, purpose-built tools:"
)

row1_col1, row1_col2 = st.columns(2)
row2_col1, row2_col2 = st.columns(2)

with row1_col1:
    with st.container(border=True):
        st.markdown("**`get_med_stock_info`** 📊")
        st.caption("Logistics & Availability")
        st.markdown("""
        * **Returns:** Status (In/Out), Quantity, Shelf Location.
        """)

with row1_col2:
    with st.container(border=True):
        st.markdown("**`get_med_regulatory_info`** ⚖️")
        st.caption("Legal & Compliance")
        st.markdown("""
        * **Returns:** Prescription status (Rx), Age limits, Narcotic (BTM) status.
        """)

with row2_col1:
    with st.container(border=True):
        st.markdown("**`get_med_safety_info`** 💊")
        st.caption("Medical Safety")
        st.markdown("""
        * **Returns:** Active ingredients, Dosages, Side effects, Contraindications.
        """)

with row2_col2:
    with st.container(border=True):
        st.markdown("**`get_med_base_info`** 📦")
        st.caption("Product Identity")
        st.markdown("""
        * **Returns:** Brand name, Form (Tablet/Syrup), Price, Pack size.
        """)


st.markdown("### ⚡ Tech Stack")

st.markdown("""
* **Database:** **Supabase** (PostgreSQL) hosts inventory and medical records.
* **Data Source:** Synthetic dataset of **16 representative medications**, generated using **Gemini 3** to ensure clean, structured test data. (📊[Database page](/database).)
* **Frontend:** Streamlit (Python UI).
* **Orchestration:** LangChain & LangGraph (Agent Loops).
* **Observability:** LangSmith (Tracing & Debugging).
* **Embeddings:** OpenAI `text-embedding-3-small`.
* **Production Model:** `gpt-5-mini`, chosen for its superior balance of speed, cost, and high F1-score.
* **Benchmark Models:** Tested alongside `gpt-4.1` and `llama-3.1-8b` for performance comparison.
""")

st.divider()

st.header("2. Evaluation Highlights")

st.write("""
To validate performance, we designed two specialized datasets (**$N=20$ for Safety**, **$N=40$ for Tool Routing**)
to rigorously benchmark the agent's adherence to safety protocols and its ability to orchestrate complex multi-tool workflows.
We were testing the following models: `gpt-4.1`, `llama-3.1-8b`, and `gpt-5-mini`.
""")

st.success("""
**🏆 Key Findings:**
* **`gpt-5-mini`** emerged as the superior model, achieving a **perfect 100% Safety Compliance** (zero unauthorized medical advice).
* In Tool Routing, it secured a dominant **98.1% F1-Score**, proving it to be the most reliable choice for navigating complex pharmacy logistics without errors.
""")

st.write("""
More information you can find on the 📊[Evaluation page](/evaluation).
""")

st.divider()

st.header("3. Limitations & Future Work")
st.write(
    "As an MVP, this system prioritizes core functionality over production readiness. To name the key limitations:"
)

st.markdown("""
* **🧠 Memory Management:** Current state is stored in-memory only and grows linearly without pruning, relying solely on the models large context window (for `gpt-5-mini` 400k context window).
* **🛡️ Missing Guardrails:** No dedicated input/output filters are implemented to strictly prevent jailbreaks or block out-of-domain queries beyond basic prompt instructions.
* **🏗️ Architecture:** The agent logic is currently embedded directly within the Streamlit frontend. For a production system, this should be decoupled into a dedicated Backend API to improve scalability and security.
""")

st.divider()
