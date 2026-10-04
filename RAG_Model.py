# Goals: Intro to ...
# - Document Retrieval
# - Augmenting LLMs with the Expert Information

# LIBRARIES

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
from langchain_chroma import Chroma
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
# from langchain_community.document_loaders import UnstructuredExcelLoader
import pandas as pd
from pprint import pprint

from langchain_ollama import OllamaEmbeddings

os.environ["OPENAI_API_KEY"] = yaml.safe_load(open('credentials.yml'))['openai']
def get_4YearPlans():
    # documents = None
    # for file in os.scandir("C:/Users/alexj/OneDrive/Desktop/Data Analytics Masters/BANA 6620/Final_RAG_Model/RAG_Model/4-Year_Plans_2024-25"):
    #     loader = Docx2txtLoader(file.path)
    #     documents = loader.load()
    loader = Docx2txtLoader("C:/Users/alexj/OneDrive/Desktop/Data Analytics Masters/BANA 6620/Final_RAG_Model/RAG_Model/4-Year_Plans_2024-25/accounting_4yp_2024-25.docx")
    documents = loader.load()
    return documents

def get_catalog():
    # documents = None
    # for file in os.scandir("C:/Users/alexj/OneDrive/Desktop/Data Analytics Masters/BANA 6620/Final_RAG_Model/RAG_Model/Business_Course_Catalog_Descriptions"):
    #     loader = Docx2txtLoader(file)
    #     documents = loader.load()
    loader = Docx2txtLoader("C:/Users/alexj/OneDrive/Desktop/Data Analytics Masters/BANA 6620/Final_RAG_Model/RAG_Model/Business_Course_Catalog_Descriptions/graduate_business_course_descriptions.docx")
    documents = loader.load()
    return documents
    return documents

# def get_templates():
#     documents = None
#     for file in os.scandir("C:/Users/alexj/OneDrive/Desktop/Data Analytics Masters/BANA 6620/Final_RAG_Model/RAG_Model/Degree Plan Templates"):
#         loader = UnstructuredExcelLoader(file.path)
#         documents = loader.load().append(documents)
#     return documents

# def get_classes():
#     documents = None
#     for file in os.scandir("C:/Users/alexj/OneDrive/Desktop/Data Analytics Masters/BANA 6620/Final_RAG_Model/RAG_Model/Fall_2005_Class_List"):
#         loader = UnstructuredExcelLoader(file.path)
#         documents = loader.load().append(documents)
#     return documents

def get_minors():
    documents = None
    for file in os.scandir("C:/Users/alexj/OneDrive/Desktop/Data Analytics Masters/BANA 6620/Final_RAG_Model/RAG_Model/Minors"):
        loader = PyPDFLoader(file.path)
        documents = loader.load()
    return documents
question = ""
def vector_chroma_db(question):
    year_plan = get_4YearPlans()
    catalog = get_catalog()
    # template = get_templates()
    # classes = get_classes()
    # minors = get_minors()

    # # Text Splitting

    # CHUNK_SIZE = 1000


    # # Recursive Character Splitter: Uses "smart" splitting, and recursively tries to split until text is small enough
    # text_splitter_recursive = RecursiveCharacterTextSplitter(
    # chunk_size = CHUNK_SIZE,
    # chunk_overlap=100,
    # )
    
    # plan__recursive = text_splitter_recursive.split_documents(year_plan)
    # catalog__recursive = text_splitter_recursive.split_documents(catalog)
    # # template__recursive = text_splitter_recursive.split_documents(template)
    # # classes__recursive = text_splitter_recursive.split_documents(classes)
    # # minors__recursive = text_splitter_recursive.split_documents(minors)

    # Text Embeddings
    embedding_function = OllamaEmbeddings(model="mxbai-embed-large")

    # * Langchain Vector Store: Chroma DB
    # https://python.langchain.com/docs/integrations/vectorstores/chroma
    documents = year_plan
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

# * LLM Specification


# new code
model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.0)

# * Combine with Lang Chain Expression Language (LCEL)
#   - Context: Give it access to the retriever
#   - Question: Provide the user question as a pass through from the invoke method
#   - Use LCEL to add a prompt template, model spec, and output parsing

rag_chain = (
    {"context": vector_chroma_db(question), "question": RunnablePassthrough()} # here is why Runnable Passthrough is needed when retriver is used
    | prompt
    | model
    | StrOutputParser()
)

result = rag_chain.invoke(
    "What is one core class for an accounting major?"
)

print(result)

# # Joins document contents into a single {text}
# def join_docs(docs):
#     return {"text": "\n\n".join(getattr(d, "page_content", str(d)) for d in docs)}

# """
# docs_2 = [
#     Document(page_content="Python is a programming language."),
#     Document(page_content="LangChain helps build LLM applications.")
# ]
# join_docs(docs_2)
# """



# data = loader.load()
# # OPENAI_API_KEY


# # 1.0 CREATE A RETRIEVER FROM THE VECTORSTORE
# # new code
# embedding_function = OpenAIEmbeddings(
#     model="text-embedding-ada-002"
# )

# # create a vector db of context data
# plans_db = 
# vectorstore = Chroma(
#     persist_directory="data/chroma_social_media_strategies_9_20_2025_openai.db",
#     embedding_function=embedding_function,
# )

# retriever = vectorstore.as_retriever()

# retriever

# # 2.0 USE THE RETRIEVER TO AUGMENT AN LLM

# # * Prompt template

# template = """Answer the question based only on the following context:
# {context}

# Question: {question}
# """

# prompt = ChatPromptTemplate.from_template(template)

# # * LLM Specification

# # old code
# # model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7, api_key=OPENAI_API_KEY)

# # new code
# model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)

# # * Combine with Lang Chain Expression Language (LCEL)
# #   - Context: Give it access to the retriever
# #   - Question: Provide the user question as a pass through from the invoke method
# #   - Use LCEL to add a prompt template, model spec, and output parsing

# rag_chain = (
#     {"context": retriever, "question": RunnablePassthrough()} # here is why Runnable Passthrough is needed when retriver is used
#     | prompt
#     | model
#     | StrOutputParser()
# )

# # * Try it out:

# result = rag_chain.invoke(
#     "What are the top 3 things needed in a good social media marketing strategy?"
# )

# pprint(result)
