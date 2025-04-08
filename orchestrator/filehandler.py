import os

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from unstructured.partition.pdf import partition_pdf
from unstructured.partition.csv import partition_csv
from langchain.schema import Document
import pandas as pdcd
import tempfile

UPLOAD_DIR = "uploads"
VECTOR_DB_DIR = "vectorstore"


embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)

try:
    vectorstore = FAISS.load_local(
        VECTOR_DB_DIR, embedding_model, allow_dangerous_deserialization=True
    )
except:
    vectorstore = None


def clean_csv(csv_path):
    try:
        df = pd.read_csv(csv_path, on_bad_lines='skip')
    except Exception as e:
        print(f"[ERROR] Failed to read CSV: {e}")
        return None

    temp_file = tempfile.NamedTemporaryFile(mode='w+', suffix=".csv", delete=False)
    df.to_csv(temp_file.name, index=False)
    return temp_file.name

def load_or_create_vectorstore():
    if os.path.exists(VECTOR_DB_DIR):
        print("[INFO] Loading existing vector store...")
        return FAISS.load_local(VECTOR_DB_DIR, embedding_model)
    else:
        print(
            "[INFO] Vector store doesn't exist yet. It will be created when you add documents."
        )
        return None


def add_pdf_to_vectorstore(pdf_path):
    print(f"[INFO] Processing {pdf_path}...")
    elements = partition_pdf(filename=pdf_path)
    documents=[]
    for element in elements:
            if element.text.strip():
                metadata = {
                    "source": os.path.basename(pdf_path),
                    "element_type": element.category,
                    "page_number": element.metadata.page_number if element.metadata else None
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
    print(f"[SUCCESS] {pdf_path} added to vector store.")


def add_csv_to_vectorstore(csv_path):
    print(f"[INFO] Processing {csv_path}...")
    cleaned_csv_path = clean_csv(csv_path)
    elements = partition_csv(filename=cleaned_csv_path)
    documents = [
           Document(
               page_content=element.text.strip(),
               metadata={
                   "source": os.path.basename(csv_path),
                   "element_type": element.category
               }
           )
           for element in elements if element.text.strip()
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


def query_vectorstore(question, k=5, allowed_types=None):
    global vectorstore
    if vectorstore is None:
        print("[ERROR] No documents have been added to the vector store yet.")
        return

    print(f"[QUERY] {question}")
    results = vectorstore.similarity_search(question, k=k)

    if allowed_types:
        results = [res for res in results if res.metadata.get("element_type") in allowed_types]

    for i, res in enumerate(results):
        print(
            f"\n--- Result {i+1} (Page: {res.metadata.get('page_number')}, Type: {res.metadata.get('element_type')}, Source: {res.metadata['source']}) ---\n"
            f"{res.page_content.strip()[:1000]}\n"
        )

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

if __name__ == "__main__":
    os.makedirs(UPLOAD_DIR, exist_ok=True)

    while True:
        print("\nOptions:")
        print("1. Upload a PDF")
        print("2. Upload a CSV")
        print("3. Search documents")
        print("4. Exit")
        choice = input("Choose an option (1/2/3/4): ")

        if choice == "1":
            pdf_driver()

        elif choice == "2":
            csv_driver()

        elif choice == "3":
            question = input("Enter your question: ")
            query_vectorstore(question, k=1, allowed_types=None)

        elif choice == "4":
            print("Exiting.")
            break


        else:
            print("Invalid option.")
