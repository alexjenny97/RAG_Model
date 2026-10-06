import os
import yaml
import warnings

# Suppress LangChain deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

# Document Loaders
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_community.document_loaders.excel import UnstructuredExcelLoader

# ==============================================================================
# 1. PATH CONFIGURATION & CREDENTIALS
# ==============================================================================
# Resolves paths relative to the directory where RAG_Model.py is located
BASE_DATA_DIR = os.path.dirname(os.path.abspath(__file__))
credentials_path = os.path.join(BASE_DATA_DIR, 'credentials.yml')

# Load OpenAI API Key from credentials.yml
if os.path.exists(credentials_path):
    with open(credentials_path, 'r') as f:
        credentials = yaml.safe_load(f)
        if credentials and 'openai' in credentials:
            os.environ["OPENAI_API_KEY"] = credentials['openai']
            print(">>> Successfully loaded OPENAI_API_KEY from credentials.yml")
        else:
            raise KeyError("Key 'openai' missing from credentials.yml!")
else:
    raise FileNotFoundError(
        f"Could not find 'credentials.yml' at: {credentials_path}\n"
        "Please ensure credentials.yml is in the same folder as RAG_Model.py."
    )

# ==============================================================================
# 2. DOCUMENT LOADING FUNCTIONS
# ==============================================================================
def get_4YearPlans():
    year_plans = []
    folder_path = os.path.join(BASE_DATA_DIR, "4-Year_Plans_2024-25")
    if os.path.exists(folder_path):
        for entry in os.scandir(folder_path):
            if entry.name.endswith(".docx"):
                loader = Docx2txtLoader(entry.path)
                docs = loader.load()
                if docs:
                    year_plans.append(docs[0].page_content)
    print(f"Loaded {len(year_plans)} 4-Year Plan document(s).")
    return year_plans

def get_catalog():
    catalog = []
    folder_path = os.path.join(BASE_DATA_DIR, "Business_Course_Catalog_Descriptions")
    if os.path.exists(folder_path):
        for entry in os.scandir(folder_path):
            if entry.name.endswith(".docx"):
                loader = Docx2txtLoader(entry.path)
                docs = loader.load()
                if docs:
                    catalog.append(docs[0].page_content)
    print(f"Loaded {len(catalog)} Catalog document(s).")
    return catalog

def get_templates():
    templates = []
    folder_path = os.path.join(BASE_DATA_DIR, "Degree Plan Templates")
    if os.path.exists(folder_path):
        for entry in os.scandir(folder_path):
            if entry.name.endswith(('.xlsx', '.xls')):
                loader = UnstructuredExcelLoader(entry.path)
                docs = loader.load()
                if docs:
                    templates.append(docs[0].page_content)
    print(f"Loaded {len(templates)} Template document(s).")
    return templates

def get_classes():
    classes = []
    folder_path = os.path.join(BASE_DATA_DIR, "Fall_2005_Class_List")
    if os.path.exists(folder_path):
        for entry in os.scandir(folder_path):
            if entry.name.endswith(('.xlsx', '.xls')):
                loader = UnstructuredExcelLoader(entry.path)
                docs = loader.load()
                if docs:
                    classes.append(docs[0].page_content)
    print(f"Loaded {len(classes)} Class list document(s).")
    return classes

def get_minors():
    minors = []
    folder_path = os.path.join(BASE_DATA_DIR, "Minors")
    if os.path.exists(folder_path):
        for entry in os.scandir(folder_path):
            if entry.name.endswith(".pdf"):
                loader = PyPDFLoader(entry.path)
                docs = loader.load()
                if docs:
                    minors.append(docs[0].page_content)
    print(f"Loaded {len(minors)} Minor document(s).")
    return minors

# ==============================================================================
# 3. CHROMA VECTOR STORE & RETRIEVER CREATION
# ==============================================================================
def vector_chroma_db(question=""):
    all_docs = []
    all_docs.extend(get_4YearPlans())
    all_docs.extend(get_catalog())
    all_docs.extend(get_templates())
    all_docs.extend(get_classes())
    all_docs.extend(get_minors())

    # Safety Check: Prevent Chroma crashing on empty documents
    if not all_docs:
        raise ValueError(
            f"No text content could be extracted from: '{BASE_DATA_DIR}'. "
            "Please check that your folder names match the function definitions."
        )

    # Text Splitting
    CHUNK_SIZE = 1000
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=100,
    )
    docs_recursive = text_splitter.create_documents(all_docs)

    # Text Embeddings & Persistent Chroma Store
    embedding_function = OpenAIEmbeddings(model="text-embedding-3-small")
    
    persist_dir = os.path.join(BASE_DATA_DIR, "data", "chroma_openai_vectorstore")
    
    vectorstore = Chroma.from_documents(
        docs_recursive, 
        embedding=embedding_function, 
        persist_directory=persist_dir,
        collection_name="cleo_advising"
    )

    return vectorstore.as_retriever(search_kwargs={"k": 5})

# ==============================================================================
# 4. EXECUTION / TEST PIPELINE
# ==============================================================================
if __name__ == "__main__":
    template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
    prompt = ChatPromptTemplate.from_template(template)
    model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.0)

    question = "What are the classes for a Human resources management major in year 3?"
    
    retriever = vector_chroma_db(question)
    docs = retriever.invoke(question)
    context_text = "\n\n".join([doc.page_content for doc in docs])

    chain = prompt | model | StrOutputParser()
    result = chain.invoke({"context": context_text, "question": question})

    print("\n--- Final Answer ---")
    print(result)