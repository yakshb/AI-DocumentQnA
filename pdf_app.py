# Import necessary modules
import pandas as pd
import streamlit as st 
from PIL import Image
from PyPDF2 import PdfReader

from langchain.embeddings import OpenAIEmbeddings, SentenceTransformerEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.memory import ConversationBufferWindowMemory
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from typing import List
import hashlib
from pathlib import Path
import tempfile

home_privacy = "We value and respect your privacy. To safeguard your personal details, we utilize the hashed value of your OpenAI API Key, ensuring utmost confidentiality and anonymity. Your API key facilitates AI-driven features during your session and is never retained post-visit. You can confidently fine-tune your research, assured that your information remains protected and private."

# Page configuration for Simple PDF App
st.set_page_config(
    page_title="Document Q&A with AI",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded"
    )

# Add this near the top of the file, after imports
def validate_openai_key(api_key: str) -> bool:
    """Validate OpenAI API key by attempting to create an embedding."""
    if not api_key:
        return False
    try:
        client = ChatOpenAI(api_key=api_key)
        # Test with a simple completion
        client.predict("test")
        return True
    except Exception as e:
        st.error(f"Invalid API key: {str(e)}")
        return False

# OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
st.sidebar.subheader("Setup")
OPENAI_API_KEY = st.sidebar.text_input("Enter Your OpenAI API Key:", type="password")
if OPENAI_API_KEY:
    if validate_openai_key(OPENAI_API_KEY):
        st.session_state['OPENAI_API_KEY'] = OPENAI_API_KEY
        st.sidebar.success("API Key validated successfully!")
    else:
        st.sidebar.error("Invalid API Key")
st.sidebar.markdown("Get your OpenAI API key [here](https://platform.openai.com/account/api-keys)")
st.sidebar.divider()
st.sidebar.subheader("Model Selection")
llm_model_options = ['gpt-4o', 'gpt-4o-2024-11-20','gpt-4']  # Add more models if available
model_select = st.sidebar.selectbox('Select LLM Model:', llm_model_options, index=0)
st.sidebar.markdown("""\n""")
temperature_input = st.sidebar.slider('Set AI Randomness / Determinism:', min_value=0.0, max_value=1.0, value=0.5)
st.sidebar.markdown("""\n""")
clear_history = st.sidebar.button("Clear conversation history")


with st.sidebar:
    st.divider()
    st.subheader("👨‍💻 Author: **Yaksh Birla**", anchor=False)
    
    st.subheader("🔗 Contact / Connect:", anchor=False)
    st.markdown(
        """
        - [Email](mailto:yb.codes@gmail.com)
        - [LinkedIn](https://www.linkedin.com/in/yakshb/)
        - [Github Profile](https://github.com/yakshb)
        - [Medium](https://medium.com/@yakshb)
        """
    )

    st.divider()
    st.write("Made with 🦜️🔗 Langchain and OpenAI LLMs")

# Move configurations to a separate section at the top
DEFAULT_MODEL = "gpt-4"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'

# Add proper type hints and error handling to core functions
def get_pdf_text(pdf_docs: List[tempfile._TemporaryFileWrapper]) -> str:
    text = []
    for pdf in pdf_docs:
        try:
            # Save uploaded file to temp location
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                tmp_file.write(pdf.read())
                tmp_file.seek(0)
                loader = PyPDFLoader(tmp_file.name)
                pages = loader.load()
                text.extend(page.page_content for page in pages)
        except Exception as e:
            st.error(f"Error processing {pdf.name}: {str(e)}")
            continue
    return "\n".join(text)

# Splits a given text into smaller chunks based on specified conditions
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        separators="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


# Generates embeddings for given text chunks and creates a vector store using FAISS
def get_vectorstore(text_chunks):
    # embeddings = OpenAIEmbeddings()
    embeddings = SentenceTransformerEmbeddings(model_name='all-MiniLM-L6-v2')
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

# Initializes a conversation chain with a given vector store
def get_conversation_chain(vectorstore):
    if 'OPENAI_API_KEY' not in st.session_state:
        st.error("Please enter a valid OpenAI API key")
        return None
        
    memory = ConversationBufferWindowMemory(memory_key='chat_history', return_message=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(
            temperature=temperature_input, 
            model_name=model_select,
            api_key=st.session_state['OPENAI_API_KEY']
        ),
        retriever=vectorstore.as_retriever(),
        get_chat_history=lambda h : h,
        memory=memory
    )
    return conversation_chain

# Improve the main UI section
def initialize_session_state():
    """Initialize all session state variables."""
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "doc_messages" not in st.session_state:
        st.session_state.doc_messages = [{"role": "assistant", "content": "Query your documents"}]
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

def main():
    initialize_session_state()
    
    # Check for API key before proceeding
    if 'OPENAI_API_KEY' not in st.session_state:
        st.warning("Please enter your OpenAI API key in the sidebar to continue.")
        return

    st.title("Document Q&A with AI")
    
    # Process file uploads
    user_uploads = st.file_uploader("Upload your files", accept_multiple_files=True)
    if user_uploads and st.button("Process Documents"):
        process_documents(user_uploads)

    # Handle chat interface
    handle_chat_interface()

    # Clear history if button clicked
    if clear_history:
        st.session_state.doc_messages = [{"role": "assistant", "content": "Query your documents"}]
        st.session_state.chat_history = []
        st.session_state.conversation = None
        st.experimental_rerun()

def process_documents(uploaded_files):
    """Process uploaded documents with proper error handling."""
    with st.spinner("Processing documents..."):
        try:
            # Reset conversation state
            st.session_state.doc_messages = [{"role": "assistant", "content": "Query your documents"}]
            st.session_state.chat_history = []
            
            raw_text = get_pdf_text(uploaded_files)
            text_chunks = get_text_chunks(raw_text)
            vectorstore = get_vectorstore(text_chunks)
            st.session_state.conversation = get_conversation_chain(vectorstore)
            st.success("Documents processed successfully!")
        except Exception as e:
            st.error(f"Error processing documents: {str(e)}")

def handle_chat_interface():
    """Handle chat interface with improved error handling."""
    # Display chat history
    for message in st.session_state.doc_messages:
        with st.chat_message(message['role']):
            st.write(message['content'])

    # Handle new messages
    if user_query := st.chat_input("Enter your query here"):
        if st.session_state.conversation is None:
            st.error("Please process a document first!")
            return

        # Add user message to chat
        st.session_state.doc_messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.write(user_query)

        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = st.session_state.conversation({"question": user_query})
                    ai_response = response["answer"]
                    st.write(ai_response)
                    st.session_state.doc_messages.append({"role": "assistant", "content": ai_response})
                except Exception as e:
                    st.error(f"Error getting response: {str(e)}")

if __name__ == "__main__":
    main()



