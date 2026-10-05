# Goals: Intro to ... 
# - Langchain Document Loaders
# - Text Embeddings
# - Vector Databases

from langchain_community.document_loaders import DataFrameLoader
from langchain_chroma import Chroma
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter

import pandas as pd
from pprint import pprint

from langchain_ollama import OllamaEmbeddings

# 1.0 DATA PREPARATION ----

loader = 

documents = loader.load()


# Text Splitting

CHUNK_SIZE = 1000


# Recursive Character Splitter: Uses "smart" splitting, and recursively tries to split until text is small enough
text_splitter_recursive = RecursiveCharacterTextSplitter(
   chunk_size = CHUNK_SIZE,
   chunk_overlap=100,
)

docs_recursive = text_splitter_recursive.split_documents(documents)

# Text Embeddings

embedding_function = OllamaEmbeddings(model="mxbai-embed-large")

# * Langchain Vector Store: Chroma DB
# https://python.langchain.com/docs/integrations/vectorstores/chroma

# Creates a sqlite database called vector_store.db
vectorstore = Chroma.from_documents(
    docs, 
    embedding=embedding_function, 
    persist_directory="data/chroma_social_media_strategies_9-20-2025"
)

vectorstore

# Similarity Search
result = vectorstore.similarity_search("How to create a social media strategy", k = 5)
len(result)
print(result[0].page_content)


