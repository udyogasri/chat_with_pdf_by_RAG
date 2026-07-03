import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from config import EMBEDDING_MODEL_NAME

@st.cache_resource
def get_embedding_model():
    """
    Initializes and returns the HuggingFace embedding model.
    Cached using Streamlit so it only loads into memory once!
    """
    print(f"Initializing embedding model: {EMBEDDING_MODEL_NAME}")
    
    # Initialize the HuggingFace embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
    )
    
    return embeddings
