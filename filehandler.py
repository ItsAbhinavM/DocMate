import os
import tempfile

import pandas as pd
import pytesseract
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community import vectorstores
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from PIL import Image
from unstructured.partition.csv import partition_csv
from unstructured.partition.doc import partition_doc
from unstructured.partition.docx import partition_docx
from unstructured.partition.image import partition_image
from unstructured.partition.pdf import partition_pdf
from unstructured.partition.xlsx import partition_xlsx

UPLOAD_DIR = "uploads"
VECTOR_DB_DIR = "vectorstore"


embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)

try:
    vectorstore = FAISS.load_local(
        VECTOR_DB_DIR, embedding_model, allow_dangerous_deserialization=True
    )
except Exception as e:
    print(e)
    print("error")
    vectorstore = None
print("vectorstore is", vectorstore)


def clean_csv(csv_path):
    try:
        df = pd.read_csv(csv_path, on_bad_lines="skip")
    except Exception as e:
        print(f"[ERROR] Failed to read CSV: {e}")
        return None

    temp_file = tempfile.NamedTemporaryFile(mode="w+", suffix=".csv", delete=False)
    df.to_csv(temp_file.name, index=False)
    return temp_file.name


def load_or_create_vectorstore():
    if os.path.exists(VECTOR_DB_DIR):
        print("[INFO] Loading existing vector store...")
        return FAISS.load_local(VECTOR_DB_DIR, embedding_model)
    else:
        print(
            "[INFO] Vector store doesn't exstringist yet. It will be created when you add documents."
        )
        return None

def get_document_stats():
    """Gather statistics about documents in the vector store."""
    import os
    import glob
    
    # Sample statistics structure
    stats = {
        "total_docs": 0,
        "doc_types": {},
        "token_counts": [],
        "avg_tokens_per_doc": 0,
        "total_tokens": 0
    }
    
    # Count documents in uploads directory
    upload_dir = "uploads/"
    if os.path.exists(upload_dir):
        for ext in [".pdf", ".csv", ".xlsx", ".doc", ".docx", ".txt"]:
            count = len(glob.glob(os.path.join(upload_dir, f"*{ext}")))
            if count > 0:
                stats["doc_types"][ext] = count
                stats["total_docs"] += count
    
    # Generate sample token counts
    import numpy as np
    if stats["total_docs"] > 0:
        stats["token_counts"] = np.random.normal(500, 200, stats["total_docs"]).tolist()
        stats["total_tokens"] = sum(stats["token_counts"])
        stats["avg_tokens_per_doc"] = stats["total_tokens"] / stats["total_docs"]
    else:
        # Add some sample data if no documents found
        stats["token_counts"] = [0]
    
    return stats

def add_pdf_to_vectorstore(pdf_path):
    print(f"[INFO] Processing {pdf_path}...")
    elements = partition_pdf(filename=pdf_path)
    documents = []
    for element in elements:
        if element.text.strip():
            metadata = {
                "source": os.path.basename(pdf_path),
                "element_type": element.category,
                "page_number": (
                    element.metadata.page_number if element.metadata else None
                ),
            }
            documents.append(Document(page_content=element.text, metadata=metadata))

    splitter = RecursiveCharacterTextSplitter(chunk_size=2500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)
    global vectorstore
    if vectorstore is None:
        print("[INFO] Creating new vector store...")
        vectorstore = FAISS.from_documents(chunks, embedding_model)
    else:
        vectorstore.add_documents(chunks)

    vectorstore.save_local(VECTOR_DB_DIR)
    print(f"[SUCCESS] {pdf_path} added to vector store.")


# def add_image_to_vectorstore(img_path):
#     print(f"[INFO] Processing {img_path}...")
def extract_text_from_image(img_path):
    try:
        image = Image.open(img_path)
        text = pytesseract.image_to_string(image)
        return text.strip()
    except Exception as e:
        print(f"[ERROR] Failed to extract text: {e}")
        return ""


def add_image_to_vectorstore(img_path):
    global vectorstore
    print(f"[INFO] Processing {img_path}...")

    if not os.path.exists(img_path):
        print(f"[ERROR] Image file not found.")
    extracted_text = extract_text_from_image(img_path)
    if not extracted_text:
        print(f"[WARNING] No text found in image, Skipping")

    elements = partition_image(filename=img_path)
    documents = []
    for element in elements:
        if element.text.strip():
            metadata = {
                "source": os.path.basename(img_path),
            }
            documents.append(Document(page_content=element.text, metadata=metadata))

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)
    global vectorstore
    if vectorstore is None:
        print("[INFO] Creating new vector store...")
        vectorstore = FAISS.from_documents(chunks, embedding_model)
    else:
        vectorstore.add_documents(chunks)

    vectorstore.save_local(VECTOR_DB_DIR)
    print(f"[SUCCESS] {img_path} added to vector store.")


def add_xlsx_to_vectorstore(xlsx_path):
    print(f"[INFO] Processing {xlsx_path}...")
    elements = partition_xlsx(filename=xlsx_path)
    documents = []
    for element in elements:
        if element.text.strip():
            metadata = {
                "source": os.path.basename(xlsx_path),
            }
            documents.append(Document(page_content=element.text, metadata=metadata))

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)
    global vectorstore
    if vectorstore is None:
        print("[INFO] Creating new vector store...")
        vectorstore = FAISS.from_documents(chunks, embedding_model)
    else:
        vectorstore.add_documents(chunks)

    vectorstore.save_local(VECTOR_DB_DIR)
    print(f"[SUCCESS] {xlsx_path} added to vector store.")


def add_doc_to_vectorstore(doc_path):
    print(f"[INFO] Processing {doc_path}...")
    elements = partition_doc(filename=doc_path)
    documents = []
    for element in elements:
        if element.text.strip():
            metadata = {
                "source": os.path.basename(doc_path),
                "element_type": element.category,
                "page_number": (
                    element.metadata.page_number if element.metadata else None
                ),
            }
            documents.append(Document(page_content=element.text, metadata=metadata))

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)
    global vectorstore
    if vectorstore is None:
        print("[INFO] Creating new vector store...")
        vectorstore = FAISS.from_documents(chunks, embedding_model)
    else:
        vectorstore.add_documents(chunks)

    vectorstore.save_local(VECTOR_DB_DIR)
    print(f"[SUCCESS] {doc_path} added to vector store.")


def add_docx_to_vectorstore(docx_path):
    print(f"[INFO] Processing {docx_path}...")
    elements = partition_docx(filename=docx_path)
    documents = []
    for element in elements:
        if element.text.strip():
            metadata = {
                "source": os.path.basename(docx_path),
                "element_type": element.category,
                "page_number": (
                    element.metadata.page_number if element.metadata else None
                ),
            }
            documents.append(Document(page_content=element.text, metadata=metadata))

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)
    global vectorstore
    if vectorstore is None:
        print("[INFO] Creating new vector store...")
        vectorstore = FAISS.from_documents(chunks, embedding_model)
    else:
        vectorstore.add_documents(chunks)

    vectorstore.save_local(VECTOR_DB_DIR)
    print(f"[SUCCESS] {docx_path} added to vector store.")


def add_csv_to_vectorstore(csv_path):
    print(f"[INFO] Processing {csv_path}...")
    cleaned_csv_path = clean_csv(csv_path)
    elements = partition_csv(filename=cleaned_csv_path)
    documents = [
        Document(
            page_content=element.text.strip(),
            metadata={
                "source": os.path.basename(csv_path),
                "element_type": element.category,
            },
        )
        for element in elements
        if element.text.strip()
    ]

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)
    global vectorstore
    if vectorstore is None:
        print("[INFO] Creating new vector store...")
        vectorstore = FAISS.from_documents(chunks, embedding_model)
    else:
        vectorstore.add_documents(chunks)

    vectorstore.save_local(VECTOR_DB_DIR)
    print(f"[SUCCESS] {csv_path} added to vector store.")

def get_all_documents():
    """Retrieve all documents from the vector store."""
    from langchain_community.vectorstores import Chroma  
    
    try:
        vectorstore = Chroma(persist_directory="./chroma_db")
        docs = vectorstore.similarity_search("", k=1000)        
        return docs
    except Exception as e:
        print(f"Error retrieving documents: {e}")
        return []


def query_vectorstore(question, k=5, allowed_types=None):
    global vectorstore
    if vectorstore is None:
        print("[ERROR] No documents have been added to the vector store yet.")
        return

    results = vectorstore.similarity_search(question, k=k)
    return results


def pdf_driver(file_path):
    if os.path.exists(file_path):
        add_pdf_to_vectorstore(file_path)
        print("added to verctorstore pdf")
    else:
        print("[ERROR] File not found.")


def csv_driver(file_path):
    if os.path.exists(file_path):
        print("added to verctorstore csv")
        add_csv_to_vectorstore(file_path)
    else:
        print("[ERROR] File not found.")


def xlsx_driver(file_path):
    # file_name = input("Enter the path to the XLSX: ").strip()
    if os.path.exists(file_path):
        add_xlsx_to_vectorstore(file_path)
    else:
        print("[ERROR] File not found.")


def image_driver(file_path):
    # file_name = input("Enter the path to the Image: ").strip()
    if os.path.exists(file_path):
        add_image_to_vectorstore(file_path)
    else:
        print("[ERROR] File not found")


def doc_driver(file_path):
    # file_name = input("Enter the path to the DOC: ").strip()
    if os.path.exists(file_path):
        add_doc_to_vectorstore(file_path)
    else:
        print("[ERROR] File not found.")


def docx_driver(file_path):
    # file_name = input("Enter the path to the DOC: ").strip()
    if os.path.exists(file_path):
        add_doc_to_vectorstore(file_path)
    else:
        print("[ERROR] File not found.")


if __name__ == "__main__":
    os.makedirs(UPLOAD_DIR, exist_ok=True)

    while True:
        print("\nOptions:")
        print("1. Upload a PDF")
        print("2. Upload a CSV")
        print("3. Upload a XLSX")
        print("4  Upload a Doc")
        print("5. Upload an Image")
        print("6. Search documents")
        print("7. Exit")
        choice = input("Choose an option (1/2/3/4): ")

        if choice == "1":
            pdf_driver("uploads/hridgsoc.pdf")
        if choice == "5":
            image_driver()
        elif choice == "2":
            csv_driver()
        elif choice == "3":
            xlsx_driver()
        elif choice == "4":
            doc_driver()
        elif choice == "6":
            question = input("Enter your question: ")
            print(query_vectorstore(question, k=5, allowed_types=None))
        elif choice == "7":
            print("Exiting....")
        else:
            print("Invalid option.")