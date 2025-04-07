import os

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

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
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    for doc in documents:
        doc.metadata["source"] = os.path.basename(pdf_path)
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


def query_vectorstore(question, k=1):
    global vectorstore
    if vectorstore is None:
        print("[ERROR] No documents have been added to the vector store yet.")
        return

    print(f"[QUERY] {question}")
    results = vectorstore.similarity_search(question, k=k)
    for i, res in enumerate(results):
        print(
            f"\n--- Result {i+1} (from {res.metadata['source']}) ---\n{res.page_content.strip()}\n"
        )


def pdf_driver():
    file_name = input("Enter the path to the PDF: ").strip()
    if os.path.exists(file_name):
        add_pdf_to_vectorstore(file_name)
    else:
        print("[ERROR] File not found.")


if __name__ == "__main__":
    os.makedirs(UPLOAD_DIR, exist_ok=True)

    while True:
        print("\nOptions:")
        print("1. Upload a PDF")
        print("2. Search documents")
        print("3. Exit")
        choice = input("Choose an option (1/2/3): ")

        if choice == "1":
            pdf_driver()

        elif choice == "2":
            question = input("Enter your question: ")
            query_vectorstore(question)

        elif choice == "3":
            print("Exiting.")
            break

        else:
            print("Invalid option.")
