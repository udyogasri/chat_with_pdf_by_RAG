import logging
import streamlit as st
from langchain_chroma import Chroma
from config import VECTORSTORE_DIR
from embeddings.embedding import get_embedding_model

logger = logging.getLogger(__name__)

@st.cache_resource
def get_vectorstore():
    """
    Initializes and returns the Chroma vector store.
    Cached using Streamlit so only ONE instance connects to the SQLite database,
    preventing file lock freezes on Windows!
    """
    logger.info("Initializing cached ChromaDB instance...")
    embeddings = get_embedding_model()
    return Chroma(
        persist_directory=VECTORSTORE_DIR,
        embedding_function=embeddings
    )

def store_chunks_in_chroma(chunks):
    """
    Takes the chunked documents and adds them to the cached ChromaDB instance.
    """
    if not chunks:
        logger.warning("No chunks provided to store in ChromaDB.")
        return None

    try:
        logger.info(f"Storing {len(chunks)} chunks in ChromaDB at: {VECTORSTORE_DIR}")
        vectorstore = get_vectorstore()
        vectorstore.add_documents(chunks)
        
        logger.info("Successfully stored chunks in vector database.")
        return vectorstore
    except Exception as e:
        logger.error(f"Error storing chunks in ChromaDB: {e}")
        return None

def get_retriever():
    """
    Returns a retriever object from the cached ChromaDB instance.
    """
    try:
        vectorstore = get_vectorstore()
        # Create a retriever that returns the top 15 most relevant chunks (smaller chunks, wider net)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 15})
        return retriever
    except Exception as e:
        logger.error(f"Error initializing retriever: {e}")
        return None
