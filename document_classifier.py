import os

import fitz  # PyMuPDF
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate


# ==========================================
# 1. CONFIG
# ==========================================
load_dotenv()


# ==========================================
# 2. SMART SAMPLING
# ==========================================
def extract_smart_sample(file_path: str) -> str:
    try:
        doc = fitz.open(file_path)
        total_pages = len(doc)
        text = ""

        if total_pages <= 6:
            pages_to_extract = list(range(total_pages))
        else:
            first_pages = [0, 1]
            mid = total_pages // 2
            mid_pages = [mid, mid + 1]
            last_pages = [total_pages - 2, total_pages - 1]

            pages_to_extract = sorted(
                list(set(first_pages + mid_pages + last_pages))
            )

        for page_num in pages_to_extract:
            text += f"\n\n--- BEGIN PAGE {page_num + 1} ---\n\n"
            text += doc[page_num].get_text()

        return text

    except Exception as e:
        print(f"Extraction error: {e}")
        return ""


# ==========================================
# 3. CLASSIFIER (MODIFIED OUTPUT ONLY)
# ==========================================
def classify_document_text(text: str) -> bool:
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.0
    )

    # 🔥 STRICT BOOLEAN OUTPUT PROMPT
    system_prompt = """
You are a document classifier.

TASK:
Determine if the document is primarily a corporate policy document.

RULES:
- Return TRUE if the document is mostly a corporate policy
- Minor inconsistencies, formatting issues, headers/footers, or small contradictions are acceptable
- Documents may contain noise from PDF extraction
- Return FALSE only if the document is clearly unrelated (e.g., resume, story, menu, invoice)

GUIDELINES:
- Look for structured sections (e.g., Scope, Policy, Definitions, Responsibilities)
- Look for formal or compliance-related language
- Ignore minor formatting or extraction issues

OUTPUT FORMAT:
Return ONLY one word:
TRUE or FALSE

Text:
{text}
"""

    prompt = ChatPromptTemplate.from_template(system_prompt)
    chain = prompt | llm

    response = chain.invoke({"text": text[:10000]})

    result = response.content.strip().upper()

    return result == "TRUE"


# ==========================================
# 4. MAIN FUNCTION (THIS IS WHAT YOU USE)
# ==========================================
def classify_document(file_path: str) -> bool:
    sample_text = extract_smart_sample(file_path)

    if not sample_text.strip():
        return False

    return classify_document_text(sample_text)