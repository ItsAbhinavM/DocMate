import pickle
import time
from langchain.vectorstores import FAISS
import dotenv
from langchain.document_loaders import PyPDFDirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import google.generativeai as palm
import concurrent.futures
import pandas as pd

# Configure the PaLM API
dotenv.load_dotenv()
google_api_key = os.getenv('PALM_KEY')

palm.configure(api_key=google_api_key)
models = [m for m in palm.list_models() if 'embedText' in m.supported_generation_methods]
model = models[0]

# Load the document and split it into chunks
try:
    loader = TextLoader("Functions/data.txt")
    documents = loader.load()
except Exception as e:
    print(f"Error loading data: {e}")
    exit()


# Split documents into chunks
def split_text(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=20, length_function=len)
    text_chunks = text_splitter.split_documents(documents)
    return text_chunks


def batch_embed_fn(text):
    embeddings_batch = []
    for text in text:
        embedding = palm.generate_embeddings(model=model, text=text)['embedding']
        embeddings_batch.append(embedding)
    return embeddings_batch


def process_batch(batch_index, df):
    print(f'Processing batch {batch_index}/{max_batch_number}')
    text_batch = df['Text'][batch_index:batch_index + batch_size].tolist()

    try:
        batch_embeddings = batch_embed_fn(text_batch)
    except Exception as e:
        print(f"Received Rate Limit Exceeded. Sleeping for 1 minute. Error: ```{e}```")
        # sleep
        time.sleep(60)
        batch_embeddings = batch_embed_fn(text_batch)

    return batch_embeddings

# Create DataFrame
df = pd.DataFrame(split_text(documents))

df = df.drop(columns=[1,
                      2])  # dropping headers 1 and 2 -> metadata ('metadata', {'source': 'Functions/data.txt'}) and type ('type', 'Document')
df.columns = ['Text']

# Create an empty list to store all embeddings
all_embeddings = []

batch_size = 5
max_batch_number = len(df['Text'])

with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
    # Submit batches to executor
    futures = [executor.submit(process_batch, i, df) for i in range(0, max_batch_number, batch_size)]

    # Gather results from completed futures
    for future in concurrent.futures.as_completed(futures):
        batch_embeddings = future.result()
        all_embeddings.extend(batch_embeddings)

with open('all_embeddings.pkl', 'wb') as file:
    pickle.dump(all_embeddings, file)

print('completed')
print(df.columns)
print(df.info())

