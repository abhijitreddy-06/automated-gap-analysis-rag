import os
import glob
from typing import List, Dict

from unstructured.partition.pdf import partition_pdf
from unstructured.chunking.title import chunk_by_title

from langchain_core.documents import Document


# ==========================================
# CONFIG
# ==========================================
TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"


# ==========================================
# ENV SETUP
# ==========================================
def setup_ocr():
    os.environ["TESSERACT_CMD"] = TESSERACT_PATH


# ==========================================
# 1. EXTRACTION
# ==========================================
def extract_pdf_elements(file_path: str):
    print(f"⏳ Extracting: {os.path.basename(file_path)}")

    if partition_pdf is None:
        raise ImportError(
            "PDF extraction requires 'unstructured[pdf]', which is not available in this environment."
        )

    try:
        elements = partition_pdf(
            filename=file_path,
            strategy="hi_res",
            infer_table_structure=True,
            extract_image_block_types=["Image"],
            extract_image_block_to_payload=True,
        )

        print(f"✅ Extracted {len(elements)} elements")

        table_count = sum(1 for el in elements if el.category == "Table")
        title_count = sum(1 for el in elements if el.category == "Title")

        print(f"   ↳ Tables: {table_count}, Titles: {title_count}")

        return elements

    except Exception as e:
        print(f"❌ Extraction failed: {e}")
        return []


def extract_from_folder(folder_path: str) -> Dict[str, list]:
    pdf_files = glob.glob(os.path.join(folder_path, "*.pdf"))

    if not pdf_files:
        print(f"❌ No PDFs found in {folder_path}")
        return {}

    print(f"📂 Found {len(pdf_files)} PDFs\n")

    extracted_data = {}

    for file_path in pdf_files:
        filename = os.path.basename(file_path)
        elements = extract_pdf_elements(file_path)

        if elements:
            extracted_data[filename] = elements

    return extracted_data


# ==========================================
# 2. FILTER BAD CHUNKS (NEW 🔥)
# ==========================================
def is_valid_chunk(text: str) -> bool:
    """
    Filters out noisy / useless chunks
    """
    if not text or len(text.strip()) < 50:
        return False

    # Too many special chars → garbage OCR
    special_ratio = sum(not c.isalnum() for c in text) / len(text)
    if special_ratio > 0.4:
        return False

    # Repeated characters (OCR noise)
    if len(set(text.strip())) < 10:
        return False

    return True


# ==========================================
# 3. CHUNKING
# ==========================================
def chunk_elements(extracted_data: Dict[str, list]) -> List[Document]:
    print("🔨 Chunking documents...\n")

    if chunk_by_title is None:
        raise ImportError(
            "PDF chunking requires 'unstructured[pdf]', which is not available in this environment."
        )

    all_docs = []
    total_chunks = 0
    filtered_out = 0

    for filename, elements in extracted_data.items():
        print(f"🧩 Chunking: {filename}")

        chunks = chunk_by_title(
            elements,
            max_characters=3000,
            new_after_n_chars=2400,
            combine_text_under_n_chars=500,
        )

        valid_chunks = 0

        for chunk in chunks:
            text = str(chunk)

            # 🔥 FILTER STEP
            if not is_valid_chunk(text):
                filtered_out += 1
                continue

            metadata = chunk.metadata.to_dict()
            metadata["source"] = filename

            doc = Document(
                page_content=text,
                metadata=metadata
            )

            all_docs.append(doc)
            valid_chunks += 1

        total_chunks += valid_chunks
        print(f"   ✅ Kept {valid_chunks} chunks")

    print("\n===================================")
    print(f"🎉 Total valid chunks: {total_chunks}")
    print(f"🚫 Filtered out: {filtered_out}")

    return all_docs


# ==========================================
# 4. STORE IN CHROMA
# ==========================================
def store_chunks(documents: List[Document], db_path: str):
    if not documents:
        print("❌ No documents to store")
        return None

    try:
        from langchain_chroma import Chroma
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError as exc:
        raise ImportError(
            "Storing chunks requires compatible 'langchain-chroma' and 'langchain-huggingface' packages."
        ) from exc

    print("🧠 Loading embedding model...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )

    print(f"💾 Storing {len(documents)} chunks in ChromaDB...")
    
    vector_db = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=db_path,
    )

    print("✅ DB created successfully")

    return vector_db


# ==========================================
# 5. FULL PIPELINE (REUSABLE)
# ==========================================
def run_ingestion_pipeline(folder_path: str, db_path: str):
    setup_ocr()

    extracted = extract_from_folder(folder_path)

    if not extracted:
        print("❌ No data extracted")
        return None

    chunked_docs = chunk_elements(extracted)

    if not chunked_docs:
        print("❌ No valid chunks after filtering")
        return None

    vector_db = store_chunks(chunked_docs, db_path)

    return vector_db


# ==========================================
# 6. OPTIONAL TEST
# ==========================================
if __name__ == "__main__":
    FOLDER = r"A:\RAG\policy"
    DB_PATH = r"A:\RAG\nexus_chroma_db_local"

    db = run_ingestion_pipeline(FOLDER, DB_PATH)

    if db:
        print("\n🔍 Test query...")
        results = db.similarity_search(
            "penalty for disabling antivirus",
            k=1
        )

        if results:
            print(f"📄 Source: {results[0].metadata.get('source')}")
            print(f"📝 Preview: {results[0].page_content[:150]}")