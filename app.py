import streamlit as st
import os
import shutil
from config import DATA_DIR, VECTORSTORE_DIR
from loaders.pdf_loader import load_and_split_pdf
from rag.retriever import store_chunks_in_chroma
from rag.chain import get_rag_chain

# Streamlit Page Configuration
st.set_page_config(page_title="Chat with PDF (RAG)", page_icon="📄")
st.title("📄 Chat with PDF using Local RAG")
st.write("Upload a PDF document and ask questions about its content. Powered by LangChain, ChromaDB, and local Ollama.")

def cleanup_directories():
    """Helper function to clear old data when a new PDF is uploaded."""
    if os.path.exists(DATA_DIR):
        for f in os.listdir(DATA_DIR):
            try:
                os.remove(os.path.join(DATA_DIR, f))
            except Exception:
                pass
    
    # Note: On Windows, ChromaDB holds a lock on the database files while Streamlit runs.
    # To prevent "Access is denied" PermissionErrors, we do not delete the VECTORSTORE_DIR.
    # Instead, new PDFs will simply be added to the existing database!
# Sidebar for file upload
with st.sidebar:
    st.header("Document Upload")
    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
    
    if uploaded_file is not None:
        # Check if we already processed this file in the current session
        if "processed_file" not in st.session_state or st.session_state.processed_file != uploaded_file.name:
            st.info("Processing document...")
            
            # Clean up previous data
            cleanup_directories()
            
            # Save uploaded file to data directory
            file_path = os.path.join(DATA_DIR, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Stage 1 & 2: Load and Split PDF
            with st.spinner("Extracting and chunking text..."):
                chunks = load_and_split_pdf(file_path)
                
            if len(chunks) == 0:
                st.error("Could not extract any text from this PDF. It might be a scanned image or empty. Please try a text-based PDF.")
                # We do not mark it as fully processed so they can try again
            else:
                # Stage 3: Generate Embeddings and Store in VectorDB
                with st.spinner("Generating embeddings and storing in ChromaDB..."):
                    store_chunks_in_chroma(chunks)
                    
                st.session_state.processed_file = uploaded_file.name
                st.success("Document processed successfully! You can now ask questions.")

# Main Chat Interface
st.header("Chat Interface")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Ask a question about your PDF..."):
    
    # Check if a document is loaded
    if "processed_file" not in st.session_state:
        st.error("Please upload a PDF document first.")
    else:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Initialize the RAG chain
                    rag_chain = get_rag_chain()
                    
                    # Invoke the chain with the user's question
                    # This triggers the retriever -> prompt formatting -> LLM pipeline
                    response = rag_chain.invoke(prompt)
                    
                    # Display the response
                    st.markdown(response)
                    
                    # Add assistant response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    st.error(f"An error occurred: {e}. Ensure Ollama is running.")
