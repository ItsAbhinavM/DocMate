from langchain_community.embeddings import HuggingFaceEmbeddings

all_MiniLM_L6_v2 = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
paraphrase_multilingual_mpnet_base_v2 = HuggingFaceEmbeddings(model_name="paraphrase-multilingual-mpnet-base-v2")
