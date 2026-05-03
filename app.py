import streamlit as st
import os
from dotenv import load_dotenv

from utils.file_handler import save_uploaded_file

# Import your pipelines
from services.policy_analysis_pipeline import run_document_pipeline
from services.rag_query_service import ask_question

# ===============================
# LOAD ENV
# ===============================
load_dotenv()

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="NexusAI",
    layout="wide"
)

st.title("🏦 NexusAI Compliance Engine")

# ===============================
# TABS
# ===============================
tab1, tab2 = st.tabs(["📄 Policy Upload", "💬 Ask Question"])

# ===============================
# TAB 1: UPLOAD + ANALYSIS
# ===============================
with tab1:
    st.subheader("Upload Policy Document")

    uploaded_file = st.file_uploader(
        "Upload a single PDF",
        type=["pdf"]
    )

    if uploaded_file:
        if st.button("Analyze Policy"):
            with st.spinner("Processing document..."):
                try:
                    # Save file
                    file_path = save_uploaded_file(uploaded_file)

                    # 🔥 IMPORTANT:
                    # Your pipeline expects file inside ./uploads folder
                    result = run_document_pipeline()

                    st.success("Analysis Complete")

                    st.markdown("### 🧾 Result")
                    st.code(result)

                except Exception as e:
                    st.error(f"Error: {e}")

# ===============================
# TAB 2: QUERY
# ===============================
with tab2:
    st.subheader("Ask Policy Question")

    query = st.text_input("Enter your question")

    if st.button("Submit Query"):
        if not query.strip():
            st.warning("Please enter a query")
        else:
            with st.spinner("Fetching answer..."):
                try:
                    DB_PATH = r"A:\RAG\nexus_chroma_db_local"

                    answer = ask_question(query, DB_PATH)

                    st.success("Answer Generated")

                    st.markdown("### 🤖 Answer")
                    st.write(answer)

                except Exception as e:
                    st.error(f"Error: {e}")