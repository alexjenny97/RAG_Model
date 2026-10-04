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

youtube_df = pd.read_csv('data/youtube_videos_9_20_25.csv')

youtube_df.head()
len(youtube_df)


# * Text Preprocessing
print(youtube_df['page_content'])
youtube_df['page_content'] = youtube_df['page_content'].str.replace('\n\n', '\n', regex=False)

# * Document Loaders
#   https://python.langchain.com/docs/integrations/document_loaders/pandas_dataframe 

loader = DataFrameLoader(youtube_df, page_content_column='page_content')

documents = loader.load()

type(documents)
type(documents[0])
documents[0]
documents[0].metadata
documents[0].page_content

pprint(documents[0].page_content)

len(documents)

# Text Splitting

CHUNK_SIZE = 1000

# Character Splitter: Splits on simple default of 
text_splitter = CharacterTextSplitter(
    chunk_size=CHUNK_SIZE, 
    # chunk_overlap=100,
    separator="\n"
)

docs = text_splitter.split_documents(documents)
len(docs)
docs[0]
docs[1]
docs[2]
docs[129]
len(docs)

# Recursive Character Splitter: Uses "smart" splitting, and recursively tries to split until text is small enough
# text_splitter_recursive = RecursiveCharacterTextSplitter(
#    chunk_size = CHUNK_SIZE,
#    chunk_overlap=100,
#)

#docs_recursive = text_splitter_recursive.split_documents(documents)

#len(docs_recursive)

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


