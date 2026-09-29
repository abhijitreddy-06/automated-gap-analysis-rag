import os
from pathlib import Path
from typing import List
from pydantic import BaseModel, Field

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser

from langchain_core.runnables import RunnablePassthrough, RunnableLambda

from langchain_core.documents import Document

from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import Chroma

from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever, ContextualCompressionRetriever

from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_classic.retrievers.document_compressors.cross_encoder_rerank import CrossEncoderReranker


def load_env():
    if not os.getenv("GOOGLE_API_KEY"):
        env_path = Path.cwd() / ".env"
        if env_path.exists():
            for line in env_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value

    if not os.getenv("GOOGLE_API_KEY"):
        raise EnvironmentError("GOOGLE_API_KEY is not set.")


class ExpandedQuery(BaseModel):
    original_query: str = Field(description="User input")
    expanded_search_term: str = Field(description="Optimized query")
    reasoning: str = Field(description="Why expanded")


def get_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.0
    )


def build_query_rewrite_chain(llm):
    parser = PydanticOutputParser(pydantic_object=ExpandedQuery)

    prompt = ChatPromptTemplate.from_template("""
    You are an expert Query Translator for a Tier-1 Bank's Policy Database.

    FOLLOW THESE STEPS:
    1. Identify the core policy topic
    2. Add industry-standard keywords
    3. Remove casual language
    4. Optimize for semantic search
    5. Keep it concise

    Return your response using this exact format:
    {format_instructions}

    User Input: {user_input}
    """)

    return (
        prompt.partial(format_instructions=parser.get_format_instructions())
        | llm
        | parser
    )


def initialize_retrievers(
    db_directory: str,
    embedding_model: str,
    reranker_model: str,
):
    print("Initializing retrievers...")

    embeddings = HuggingFaceBgeEmbeddings(
        model_name=embedding_model,
        encode_kwargs={"normalize_embeddings": True},
    )

    vector_db = Chroma(
        persist_directory=db_directory,
        embedding_function=embeddings,
    )

    vector_retriever = vector_db.as_retriever(search_kwargs={"k": 10})

    print("   Loading BM25 index...")
    db_data = vector_db.get()

    if not db_data["documents"]:
        raise ValueError("No documents found in Chroma DB")

    documents = [
        Document(page_content=txt, metadata=meta)
        for txt, meta in zip(db_data["documents"], db_data["metadatas"])
    ]

    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = 10

    ensemble = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.4, 0.6],
    )

    print("   Loading reranker...")
    cross_encoder = HuggingFaceCrossEncoder(model_name=reranker_model)

    compressor = CrossEncoderReranker(
        model=cross_encoder,
        top_n=3,
    )

    compression_retriever = ContextualCompressionRetriever(
        base_retriever=ensemble,
        base_compressor=compressor,
    )

    return compression_retriever


def retrieve_documents(retriever, query: str):
    print(f"Retrieval Query:\n{query}\n")
    return retriever.invoke(query)

def format_documents(docs: List[Document]) -> str:
    if not docs:
        return "NO_RELEVANT_POLICIES_FOUND"

    formatted = []
    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "Unknown Document")
        formatted.append(f"--- [Document {i}: {source}] ---\n{doc.page_content}\n")

    return "\n".join(formatted)


def build_rag_chain(llm, query_chain, retriever):
    synthesis_prompt = ChatPromptTemplate.from_template("""
    You are a strict, highly accurate Compliance Officer for Nexus Bank.
    Answer using ONLY the provided context.

    RULES:
    1. If no answer → "I cannot answer this based on the current policies."
    2. No hallucination
    3. Always cite document name

    Context:
    {context}

    Question: {question}
    """)

    extract_query = RunnableLambda(lambda x: x.expanded_search_term)

    retrieval_chain = (
        query_chain
        | extract_query
        | RunnableLambda(lambda q: retrieve_documents(retriever, q))
    )

    final_chain = (
        {
            "context": retrieval_chain | RunnableLambda(format_documents),
            "question": RunnablePassthrough(),
        }
        | synthesis_prompt
        | llm
        | StrOutputParser()
    )

    return final_chain


def ask_question(query: str, db_path: str) -> str:
    load_env()

    llm = get_llm()

    query_chain = build_query_rewrite_chain(llm)

    retriever = initialize_retrievers(
        db_directory=db_path,
        embedding_model="BAAI/bge-large-en-v1.5",
        reranker_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
    )

    rag_chain = build_rag_chain(llm, query_chain, retriever)

    return rag_chain.invoke(query)


if __name__ == "__main__":
    DB_PATH = r"A:\RAG\nexus_chroma_db_local"

    query = "is screen recording allowed on office laptops?"

    print("User:", query)
    print("Processing...")

    answer = ask_question(query, DB_PATH)

    print("ANSWER:")
    print("=" * 60)
    print(answer)
    print("=" * 60)