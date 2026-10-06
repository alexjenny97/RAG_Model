# Module Imports
import streamlit as st
import pandas as pd

# 1. Decouple backend loading with Streamlit Caching
try:
    import rag_backend
    import importlib
    importlib.reload(rag_backend)
    
    from rag_backend import get_vector_store, query_rag_system
    print("\n>>> SUCCESS: Imported real rag_backend! <<<\n")
except Exception as err:
    print(f"\n>>> IMPORT ERROR: {err} <<<\n")
    import_error_message = str(err)
    def query_rag_system(query: str, vectorstore=None) -> str:
        return f"⚠️ **Import Error in `rag_backend.py`:**\n```\n{import_error_message}\n```"
    def get_vector_store():
        return None

# ==============================================================================
# 1. PAGE CONFIGURATION & BRANDING
# ==============================================================================
st.set_page_config(
    page_title="Cleo | CU Denver Business School's Academic Advising Assistant",
    page_icon="🐾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Cleo branding aligned with CU Denver Brand Guidelines
st.markdown("""
    <style>
    /* CU Denver Brand Palette */
    :root {
        --cu-gold: #CFB87C;
        --cu-accessible-gold: #8D7334;
        --cu-sandstone: #F1EAD8;
        --cu-dark-gray: #565A5C;
        --cu-black: #1A1A1A;
    }

    /* Main Container Padding */
    .block-container {
        padding-top: 5rem;
        padding-bottom: 5rem;
    }

    /* Main Headers */
    .main-header {
        font-size: 2.3rem;
        font-weight: 800;
        color: var(--cu-black);
        margin-bottom: 0px;
        letter-spacing: -0.5px;
    }
    
    .gold-accent {
        color: var(--cu-accessible-gold);
    }

    .sub-header {
        font-size: 1.05rem;
        color: var(--cu-dark-gray);
        margin-bottom: 25px;
        font-weight: 400;
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #CFB87C;
        border-right: 2px solid var(--cu-sandstone);
    }

    /* Chat Messages */
    .stChatMessage {
        border-radius: 8px;
        padding: 12px;
        margin-bottom: 8px;
    }

    /* User Message Styling */
    [data-testid="stChatMessage"]:nth-child(even) {
        background-color: var(--cu-sandstone) !important;
        border-left: 4px solid var(--cu-gold);
    }

    /* Primary Buttons (Clear Chat) */
    .stButton>button {
        background-color: var(--cu-black);
        color: #FFFFFF;
        border: 1px solid var(--cu-black);
        border-radius: 6px;
        font-weight: 600;
        transition: all 0.2s ease;
    }
    
    .stButton>button:hover {
        background-color: var(--cu-gold);
        color: var(--cu-black);
        border-color: var(--cu-gold);
    }

    /* Input Focus Border */
    .stTextInput>div>div>input:focus {
        border-color: var(--cu-gold) !important;
        box-shadow: 0 0 0 1px var(--cu-gold) !important;
    }
    </style>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. CACHED VECTOR STORE INITIALIZATION
# ==============================================================================
@st.cache_resource
def load_cached_vectorstore():
    """
    Caches the Chroma DB connection in memory so Streamlit doesn't 
    re-load embeddings and database connections on every turn.
    """
    return get_vector_store()

vectorstore = load_cached_vectorstore()

# ==============================================================================
# 3. SIDEBAR NAVIGATION & BENCHMARKS
# ==============================================================================
with st.sidebar:
    st.title("Meet Cleo!")
    st.markdown("""
    **CU Denver Business School's Academic AI Assistant**  
    *Milo the Lynx's very organized Academic Advising cousin. Cleo digitized the entire CU Denver Business advising catalog so you can get all the answers you need to plan your course schedules!*
    
    Equipped with LangChain, ChromaDB, and `llama3.2`, Cleo analyzes academic advising data to provide fast, deterministic answers!
    """)
    
    st.divider()
    
    # Quick Test Queries Sidebar Helper
    st.subheader("💡 Sample Test Queries")
    sample_queries = [
        "What are the core courses in Marketing?",
        "What are the graduation requirements for Accounting?",
        "What is the course information for BANA 6620?",
        "What are the time and location for BANA 6620 in fall 2025?",
        "What courses does Ziyi Wang teach in Fall 2025, and what is his email address?",
        "What is the prerequisite for BANA 6620?",
        "What courses should a freshman in Marketing take in Semester 1?"
    ]
    
    selected_sample = st.selectbox("Select a benchmark query:", ["-- Select a query --"] + sample_queries)
    
    st.divider()
    if st.button("🧹 Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ==============================================================================
# 4. MAIN INTERFACE HEADER & HISTORY
# ==============================================================================
st.markdown('<p class="main-header">🐾 CU Denver Business School\'s Academic Advising Assistant</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Ask Cleo anything about course schedules, degree requirements, prerequisites, or faculty info.</p>', unsafe_allow_html=True)

# Initialize Session State for Chat History
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! I'm Cleo, your CU Denver academic advisor assistant. How can I help you plan your courses today?"}
    ]

# Render existing chat messages
for message in st.session_state.messages:
    avatar = "🐾" if message["role"] == "assistant" else "🎓"
    with st.chat_message(message["role"], avatar=avatar):
        content = message.get("content")
        if content is None:
            st.write("*(No response text)*")
        else:
            st.markdown(content)

# ==============================================================================
# 5. CHAT LOGIC & CACHED RAG INTEGRATION
# ==============================================================================
prompt_input = st.chat_input("Ask Cleo a question...")

# Check if a benchmark query was chosen from the sidebar
if selected_sample != "-- Select a query --" and not prompt_input:
    prompt_input = selected_sample

# Process submitted prompt
if prompt_input:
    st.session_state.messages.append({"role": "user", "content": prompt_input})
    with st.chat_message("user", avatar="🎓"):
        st.markdown(prompt_input)

    with st.chat_message("assistant", avatar="🐾"):
        with st.spinner("Cleo is searching CU Denver Business documents..."):
            try:
                # Pass cached vectorstore directly into query_rag_system
                response_text = query_rag_system(prompt_input, vectorstore=vectorstore)
                if not response_text:
                    response_text = "⚠️ **Backend Error:** `query_rag_system` returned empty or `None`."
            except Exception as e:
                response_text = f"⚠️ **Error connecting to backend:** {str(e)}"
            
            st.markdown(response_text)

    st.session_state.messages.append({"role": "assistant", "content": response_text})