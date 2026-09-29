import os
import glob
from pathlib import Path
from typing import List

from document_classifier import classify_document
from ingestion_service import (
    extract_pdf_elements,
    chunk_elements,
)
from services.rag_query_service import (
    load_env,
    get_llm,
    initialize_retrievers,
    retrieve_documents,
    format_documents,
)

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

UPLOAD_FOLDER = "./uploads"
DB_PATH = r"A:\RAG\nexus_chroma_db_local"

def get_single_pdf(upload_folder: str) -> str:
    pdf_files = glob.glob(os.path.join(upload_folder, "*.pdf"))

    if not pdf_files:
        raise FileNotFoundError("No PDF found")

    if len(pdf_files) > 1:
        raise ValueError("Only one PDF allowed")

    return pdf_files[0]

def process_document(file_path: str) -> List[Document]:
    filename = os.path.basename(file_path)

    print(f"Extracting + Chunking: {filename}")

    elements = extract_pdf_elements(file_path)

    if not elements:
        return []

    chunked_docs = chunk_elements({filename: elements})

    return chunked_docs

def build_gap_analysis_prompt():
    return ChatPromptTemplate.from_template("""
You are a strict banking compliance analyst.

You are given:
1. A NEW policy document
2. Existing policy context retrieved from database

TASK:
Decide ONE of the following:

CASE 1: COMPLETELY NEW
- If new document is unrelated to existing policies
- Output:
NEW_POLICY
Summary: <short summary>

CASE 2: PARTIALLY SIMILAR
- If overlaps exist but differences are present
- Output:
GAP_ANALYSIS
Summary: <short summary>
Gaps:
- <gap 1>
- <gap 2>

CASE 3: NOT ENOUGH INFO
- If context insufficient
- Output:
INSUFFICIENT_DATA

STRICT RULES:
- Do NOT hallucinate
- Use ONLY given context
- Be concise

----------------------

NEW DOCUMENT:
{new_doc}

----------------------

EXISTING CONTEXT:
{context}
""")


def run_gap_analysis(new_docs: List[Document], retriever):
    llm = get_llm()

    new_text = "\n\n".join([doc.page_content for doc in new_docs])[:8000]

    query = new_docs[0].page_content[:300]

    retrieved_docs = retrieve_documents(retriever, query)

    context = format_documents(retrieved_docs)

    prompt = build_gap_analysis_prompt()
    chain = prompt | llm

    response = chain.invoke({
        "new_doc": new_text,
        "context": context
    })

    return response.content

def run_document_pipeline():
    load_env()

    file_path = get_single_pdf(UPLOAD_FOLDER)
    filename = os.path.basename(file_path)

    print(f"Processing: {filename}")

    is_policy = classify_document(file_path)

    if not is_policy:
        print("Not a policy document. Stopping.")
        return "REJECTED"

    print("Policy detected. Proceeding...")

    chunked_docs = process_document(file_path)

    if not chunked_docs:
        print("No usable content")
        return "FAILED"

    retriever = initialize_retrievers(
        db_directory=DB_PATH,
        embedding_model="BAAI/bge-large-en-v1.5",
        reranker_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
    )

    result = run_gap_analysis(chunked_docs, retriever)

    print("FINAL RESULT:")
    print("=" * 60)
    print(result)
    print("=" * 60)

    return result

if __name__ == "__main__":
    run_document_pipeline()