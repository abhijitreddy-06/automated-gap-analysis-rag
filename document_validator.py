import os
import glob
from pathlib import Path

from document_classifier import classify_document

def load_env():
    env_path = Path.cwd() / ".env"
    
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            
            k, v = line.split("=", 1)
            k = k.strip()
            v = v.strip().strip('"').strip("'")

            if k and k not in os.environ:
                os.environ[k] = v

def process_single_upload(upload_folder: str = "./uploads"):
    pdf_files = glob.glob(os.path.join(upload_folder, "*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(f"No PDF found in '{upload_folder}'")

    if len(pdf_files) > 1:
        raise ValueError(
            f"Multiple PDFs found in '{upload_folder}'. Only one file is allowed."
        )

    file_path = pdf_files[0]
    filename = os.path.basename(file_path)

    print(f"Processing: {filename}")

    try:
        is_policy = classify_document(file_path)

        if is_policy:
            print("POLICY DOCUMENT")
        else:
            print("NOT A POLICY")

        return is_policy

    except Exception as e:
        print(f"Error processing {filename}: {e}")
        return False

if __name__ == "__main__":
    load_env()

    if not os.getenv("GOOGLE_API_KEY"):
        raise EnvironmentError(
            "GOOGLE_API_KEY not set. Add it to .env or environment variables."
        )

    result = process_single_upload()