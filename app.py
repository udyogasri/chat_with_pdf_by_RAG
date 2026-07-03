import streamlit as st
import os
import logging
from langchain_core.messages import HumanMessage, AIMessage
from config import DATA_DIR, VECTORSTORE_DIR
from loaders.pdf_loader import load_and_split_pdf
from rag.retriever import store_chunks_in_chroma
from rag.chain import get_rag_chain

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Streamlit Page Configuration
st.set_page_config(page_title="Chat with PDF (RAG)", page_icon="📄")
st.title("📄 Chat with PDF using Local RAG")
st.write("Upload one or more PDF documents and ask questions about their content. Powered by LangChain, ChromaDB, and local Ollama.")

def cleanup_directories():
    """Helper function to clear old data when a new session starts."""
    if os.path.exists(DATA_DIR):
        for f in os.listdir(DATA_DIR):
            try:
                os.remove(os.path.join(DATA_DIR, f))
            except Exception:
                pass

# Sidebar for file upload
with st.sidebar:
    st.header("Document Upload")
    # 1. Multiple PDF Support: accept_multiple_files=True
    uploaded_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)
    
    if st.button("Clear Vector Database & Start Over"):
        st.session_state.processed_files = set()
        st.session_state.messages = []
        cleanup_directories()
        st.success("Data cleared! Please upload files again.")

    if "processed_files" not in st.session_state:
        st.session_state.processed_files = set()
        # Restore session state from the data directory on page refresh
        if os.path.exists(DATA_DIR):
            for f in os.listdir(DATA_DIR):
                if f.endswith(".pdf"):
                    st.session_state.processed_files.add(f)
                    
    # Display currently loaded resumes so the user knows they are there even after a refresh
    if st.session_state.processed_files:
        st.markdown("### 📚 Currently Loaded Resumes")
        for f in sorted(st.session_state.processed_files):
            st.markdown(f"- {f}")

    if uploaded_files:
        new_files_to_process = [f for f in uploaded_files if f.name not in st.session_state.processed_files]
        
        if new_files_to_process:
            with st.spinner(f"Processing {len(new_files_to_process)} new document(s)..."):
                for uploaded_file in new_files_to_process:
                    # Save uploaded file to data directory
                    file_path = os.path.join(DATA_DIR, uploaded_file.name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    # Stage 1 & 2: Load and Split PDF
                    chunks = load_and_split_pdf(file_path)
                        
                    if not chunks:
                        st.error(f"Could not extract text from {uploaded_file.name}. It might be scanned or empty.")
                    else:
                        # Stage 3: Generate Embeddings and Store in VectorDB
                        store_chunks_in_chroma(chunks)
                        st.session_state.processed_files.add(uploaded_file.name)
                        
            st.success(f"Successfully processed {len(new_files_to_process)} new document(s)!")

# Main Chat Interface
st.header("Chat Interface")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for i, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            for j, source_path in enumerate(message["sources"]):
                if os.path.exists(source_path):
                    with open(source_path, "rb") as f:
                        file_name = os.path.basename(source_path)
                        st.download_button(
                            label=f"📥 Download {file_name}",
                            data=f.read(),
                            file_name=file_name,
                            mime="application/pdf",
                            key=f"download_{i}_{j}"
                        )

# Accept user input
if prompt := st.chat_input("Ask a question about your PDF(s)..."):
    
    # Check if a document is loaded
    if not st.session_state.get("processed_files"):
        st.error("Please upload at least one PDF document first.")
    else:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Initialize the RAG chain
                    rag_chain = get_rag_chain()
                    
                    if not rag_chain:
                        st.error("Failed to initialize the RAG chain. Please check the logs.")
                    else:
                        # 2. Conversation History Setup
                        # Map Streamlit's simple dict history to LangChain's HumanMessage and AIMessage objects
                        chat_history = []
                        for msg in st.session_state.messages[:-1]: # exclude the latest prompt we just appended
                            if msg["role"] == "user":
                                chat_history.append(HumanMessage(content=msg["content"]))
                            else:
                                chat_history.append(AIMessage(content=msg["content"]))
                        
                        # 3. Invoke Chain with History
                        logger.info(f"Invoking chain with prompt: {prompt}")
                        response = rag_chain.invoke({
                            "input": prompt,
                            "chat_history": chat_history
                        })
                        
                        answer = response.get("answer", "I don't know.")
                        context = response.get("context", [])
                        
                        # 4. Display the Top Source Document
                        source_files = []
                        if context:
                            # The first document in the context is mathematically the highest-ranked match.
                            # To prevent UI clutter and ensure the button always appears, we only show the absolute best match.
                            top_doc = context[0]
                            source_path = top_doc.metadata.get("source", "Unknown")
                            source_name = os.path.basename(source_path)
                            page = top_doc.metadata.get("page", 0) + 1
                            
                            answer += f"\n\n**Best Match:** {source_name} (Page {page})\n"
                            
                            if os.path.exists(source_path):
                                source_files.append(source_path)
                        
                        msg_idx = len(st.session_state.messages)
                        
                        # Add assistant response to chat history
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": answer,
                            "sources": source_files
                        })
                        
                        # Display the final combined answer with sources
                        st.markdown(answer)
                        
                        # Display download buttons for the found resumes
                        for j, source_path in enumerate(source_files):
                            with open(source_path, "rb") as f:
                                file_name = os.path.basename(source_path)
                                st.download_button(
                                    label=f"📥 Download {file_name}",
                                    data=f.read(),
                                    file_name=file_name,
                                    mime="application/pdf",
                                    key=f"download_{msg_idx}_{j}"
                                )
                except Exception as e:
                    logger.error(f"Error during chat generation: {e}")
                    st.error(f"An error occurred: {e}. Ensure Ollama is running.")
