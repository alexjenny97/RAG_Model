from langchain_chroma import Chroma

from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

import pandas as pd
import yaml
from pprint import pprint
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import DataFrameLoader
from langchain_community.document_loaders import  Docx2txtLoader
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.document_loaders.excel import UnstructuredExcelLoader
from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from msoffcrypto import OfficeFile
os.environ["OPENAI_API_KEY"] = yaml.safe_load(open('credentials.yml'))['openai']
question = ""
def get_4YearPlans():
    year_plans = []
    for file in os.scandir("C:/Users/alexj/OneDrive/Desktop/Data Analytics Masters/BANA 6620/Final_RAG_Model/RAG_Model/4-Year_Plans_2024-25"):
        loader = Docx2txtLoader("C:/Users/alexj/OneDrive/Desktop/Data Analytics Masters/BANA 6620/Final_RAG_Model/RAG_Model/4-Year_Plans_2024-25/" + str(file.name))
        documents = loader.load()
        year_plans.append(documents[0].page_content)
    return year_plans

def get_catalog():
    catalog = []
    for file in os.scandir("C:/Users/alexj/OneDrive/Desktop/Data Analytics Masters/BANA 6620/Final_RAG_Model/RAG_Model/Business_Course_Catalog_Descriptions"):
        loader = Docx2txtLoader("C:/Users/alexj/OneDrive/Desktop/Data Analytics Masters/BANA 6620/Final_RAG_Model/RAG_Model/Business_Course_Catalog_Descriptions/" + str(file.name))
        documents = loader.load()
        catalog.append(documents[0].page_content)
    return catalog

def get_templates():
    templates = []
    for file in os.scandir("C:/Users/alexj/OneDrive/Desktop/Data Analytics Masters/BANA 6620/Final_RAG_Model/RAG_Model/Degree Plan Templates"):
        loader = UnstructuredExcelLoader("C:/Users/alexj/OneDrive/Desktop/Data Analytics Masters/BANA 6620/Final_RAG_Model/RAG_Model/Degree Plan Templates/" + str(file.name))
        documents = loader.load()
        templates.append(documents[0].page_content)
    return templates

def get_classes():
    classes = []
    for file in os.scandir("C:/Users/alexj/OneDrive/Desktop/Data Analytics Masters/BANA 6620/Final_RAG_Model/RAG_Model/Fall_2005_Class_List"):
        loader = UnstructuredExcelLoader("C:/Users/alexj/OneDrive/Desktop/Data Analytics Masters/BANA 6620/Final_RAG_Model/RAG_Model/Fall_2005_Class_List/" + str(file.name))
        documents = loader.load()
        classes.append(documents[0].page_content)
    return classes

def get_minors():
    minors = []
    for file in os.scandir("C:/Users/alexj/OneDrive/Desktop/Data Analytics Masters/BANA 6620/Final_RAG_Model/RAG_Model/Minors"):
        loader = PyPDFLoader("C:/Users/alexj/OneDrive/Desktop/Data Analytics Masters/BANA 6620/Final_RAG_Model/RAG_Model/Minors/" + str(file.name))
        documents = loader.load()
        minors.append(documents[0].page_content)
    return minors

def vector_chroma_db(question):
    year_plan = get_4YearPlans()
    catalog = get_catalog()
    all_docs = year_plan
    for x in catalog:
        all_docs.append(x)
    template = get_templates()
    print("year plan and catalog done")
    for x in template:
        all_docs.append(x)
    classes = get_classes()
    print("template done")
    for x in classes:
        all_docs.append(x)
    print("classes done")

    # minors = get_minors()
    # for x in minors:
    #             all_docs.append(x)

    # # Text Splitting

    CHUNK_SIZE = 1000
    # Recursive Character Splitter: Uses "smart" splitting, and recursively tries to split until text is small enough
    text_splitter_recursive = RecursiveCharacterTextSplitter(
    chunk_size = CHUNK_SIZE,
    chunk_overlap=100,
    )
    docs__recursive = text_splitter_recursive.create_documents(all_docs)
    # Text Embeddings
    embedding_function = OllamaEmbeddings(model="mxbai-embed-large")

    documents = docs__recursive
    # Creates a sqlite database called vector_store.db
    vectorstore = Chroma.from_documents(
        documents, 
        embedding=embedding_function, 
        persist_directory="data/chroma_social_media_strategies_9-20-2025"
    )


    # Similarity Search
    result = vectorstore.similarity_search(question, k = 5)
    retriever = vectorstore.as_retriever()

    return retriever

#  2.0 USE THE RETRIEVER TO AUGMENT AN LLM

# * Prompt template
template = """Answer the question based only on the following context:
{context}

Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

# new code
model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.0)

rag_chain = (
    {"context": vector_chroma_db(question), "question": RunnablePassthrough()} # here is why Runnable Passthrough is needed when retriver is used
    | prompt
    | model
    | StrOutputParser()
)

result = rag_chain.invoke(
    "What are the classes for a Human resources management major in year 3?"
)

print(result)