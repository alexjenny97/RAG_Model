import os
import yaml
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ==============================================================================
# 1. PATH CONFIGURATION & CREDENTIALS
# ==============================================================================
BASE_DATA_DIR = os.path.dirname(os.path.abspath(__file__))
credentials_path = os.path.join(BASE_DATA_DIR, 'credentials.yml')

# Ensure OpenAI API Key is loaded
if os.path.exists(credentials_path):
    with open(credentials_path, 'r') as f:
        credentials = yaml.safe_load(f)
        if credentials and 'openai' in credentials:
            os.environ["OPENAI_API_KEY"] = str(credentials['openai']).strip()
        else:
            raise KeyError("Key 'openai' missing from credentials.yml!")
else:
    raise FileNotFoundError(f"Could not find 'credentials.yml' at: {credentials_path}")

# Persistent directory for OpenAI vectorstore
PERSIST_DIR = os.path.join(BASE_DATA_DIR, "data", "chroma_openai_vectorstore")

# ==============================================================================
# 2. VECTOR STORE & RAG BACKEND FUNCTIONS
# ==============================================================================
def get_vector_store():
    """
    Loads and returns the persistent Chroma vector store created by RAG_Model.py.
    """
    if not os.path.exists(PERSIST_DIR):
        raise FileNotFoundError(
            f"Vector database not found at {PERSIST_DIR}. "
            "Please run 'python RAG_Model.py' first to build and save the database."
        )

    embedding_function = OpenAIEmbeddings(model="text-embedding-3-small")
    return Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embedding_function,
        collection_name="cleo_advising"
    )

def query_rag_system(question: str, vectorstore=None) -> str:
    """
    Queries the RAG system with a user question and returns the generated answer.
    Accepts an optional cached vectorstore instance.
    """
    if vectorstore is None:
        vectorstore = get_vector_store()

    #top-k similarity search on user query to return to 5 most similar document chunks from database
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    docs = retriever.invoke(question)
    context_text = "\n\n".join([doc.page_content for doc in docs])

    #system prompt grounding Cleo's persona
    template = """You are Cleo, an academic advisor assistant for CU Denver Business School. 
Answer the question based only on the following context:
{context}

Question: {question}
"""
    prompt = ChatPromptTemplate.from_template(template)
    model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.0)
    chain = prompt | model | StrOutputParser() #LCEL Pipeline Execution

    return chain.invoke({"context": context_text, "question": question})

if __name__ == "__main__":
    # Test query
    sample_q = "What are the classes for a Human resources management major in year 3?"
    print("Testing backend...")
    print("Answer:", query_rag_system(sample_q))