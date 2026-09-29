import streamlit as st
import os
from dotenv import load_dotenv

from utils.file_handler import save_uploaded_file

from services.policy_analysis_pipeline import run_document_pipeline
from services.rag_query_service import ask_question

load_dotenv()

st.set_page_config(
    page_title="NexusAI",
    layout="wide"
)

st.title("NexusAI Compliance Engine")

tab1, tab2 = st.tabs(["Policy Upload", "Ask Question"])

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
                    file_path = save_uploaded_file(uploaded_file)
                    result = run_document_pipeline()
                    st.success("Analysis Complete")
                    st.markdown("### Result")
                    st.code(result)
                except Exception as e:
                    st.error(f"Error: {e}")

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
                    st.markdown("### Answer")
                    st.write(answer)
                except Exception as e:
                    st.error(f"Error: {e}")