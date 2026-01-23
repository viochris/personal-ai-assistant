import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

def init_state():
    """
    Initializes all necessary session state variables at the start of the application.
    This prevents 'KeyError' exceptions when accessing state variables later.
    """
    if "messages" not in st.session_state:
        st.session_state.messages = [] # Holds the chat display messages
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [] # Holds the conversation context for the LLM
    if "qa_chain" not in st.session_state:
        st.session_state.qa_chain = None # Holds the LangChain Conversational Chain object
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None # Holds the FAISS vector database object

def reset_state():
    """
    Callback function for the 'Reset Conversation' button.
    It clears the conversation history but retains the loaded Vector Store and QA Chain
    to ensure the user does not need to reload the data.
    """
    st.session_state.messages = []
    st.session_state.chat_history = []
    
    # Provide visual feedback that the action was successful
    st.toast("Conversation history cleared!", icon="🧹")

def change_on_api_key():
    """
    Callback function triggered when the API Key input changes.
    Performs a HARD RESET on all state variables (including Vector Store and Chain)
    to ensure security and re-authentication with the new credentials.
    """
    st.session_state.messages = []
    st.session_state.chat_history = []
    st.session_state.qa_chain = None
    st.session_state.vectorstore = None
    
    # Alert the user that the system memory has been completely reset
    st.toast("API Key updated. System memory reset!", icon="🔑")

def change_on_lan():
    """
    Callback function triggered when the Language selection changes.
    It resets the 'qa_chain' to force a reconstruction of the chain with the 
    new language-specific system prompt. The Vector Store is preserved.
    """
    st.session_state.qa_chain = None
    
    # Notify the user that the language settings are being applied
    st.toast("Language settings updated! Applying changes...", icon="🌐")

def load_css():
    """
    Injects custom CSS styles into the Streamlit app.
    This function handles responsive design (Media Queries) to ensure the UI 
    looks good on Laptops, Tablets, and Mobile devices.
    """
    # unsafe_allow_html=True is required to render raw CSS styles in Streamlit
    return st.markdown("""
        <style>
        /* --- GLOBAL STYLES (Default / Laptop) --- */
        
        .main-title {
            text-align: center;
            font-weight: 800;
            background: -webkit-linear-gradient(45deg, #FF4B4B, #FF914D);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 10px;
            /* Default Size for Laptops */
            font-size: 3.5rem !important; 
        }
        
        .description-box {
            background-color: var(--secondary-background-color);
            color: var(--text-color);
            border-radius: 15px;
            border: 1px solid rgba(128, 128, 128, 0.2);
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            text-align: center;
            margin: 0 auto 30px auto; /* Center the box */
            
            /* Default Spacing for Laptops */
            padding: 40px;
            font-size: 1.2rem;
            max-width: 800px; /* Prevent it from getting too wide on huge screens */
        }

        .highlight {
            color: #FF4B4B;
            font-weight: bold;
        }

        /* --- TABLET ADJUSTMENTS (Screens smaller than 768px) --- */
        @media (max-width: 768px) {
            .main-title {
                font-size: 2.5rem !important; /* Slightly smaller title */
            }
            .description-box {
                padding: 25px; /* Less padding */
                font-size: 1.1rem; /* Slightly smaller text */
            }
        }

        /* --- MOBILE ADJUSTMENTS (Screens smaller than 480px) --- */
        @media (max-width: 480px) {
            .main-title {
                font-size: 2rem !important; /* Compact title for phones */
            }
            .description-box {
                padding: 15px; /* Compact padding */
                font-size: 1rem; /* Readable text for mobile */
                border-radius: 10px;
            }
        }
        </style>
    """, unsafe_allow_html=True)

def chunking_document():
    """
    Loads the profile JSON file and splits it into smaller text chunks 
    for vector embedding.
    """
    # 1. Load the raw data using TextLoader to treat JSON as a string
    loader = TextLoader("data/silvio_profile.json", encoding="utf-8")
    docs = loader.load()

    # 2. Initialize the splitter
    # Splitting by character count to maintain context in small fragments
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    
    # 3. Execute the splitting process
    text_chunks = text_splitter.split_documents(docs)

    return text_chunks