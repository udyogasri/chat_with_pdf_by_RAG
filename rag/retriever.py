from langchain_chroma import Chroma
from config import VECTORSTORE_DIR
from embeddings.embedding import get_embedding_model

def store_chunks_in_chroma(chunks):
    """
    Takes the chunked documents, generates embeddings for them, and stores them in ChromaDB.
    
    Why Vector Databases?
    Traditional SQL databases search for exact keyword matches.
    Vector databases (like ChromaDB) store the dense number arrays (embeddings). 
    This allows us to perform "Similarity Search". We can compare the numbers to find 
    text that is *semantically* similar, even if the exact keywords don't match.
    
    Args:
        chunks (list): List of LangChain Document objects (our chunked text).
        
    Returns:
        Chroma: The initialized vector store object.
    """
    print(f"Initializing embedding model for vector store...")
    embeddings = get_embedding_model()
    
    print(f"Storing {len(chunks)} chunks in ChromaDB at: {VECTORSTORE_DIR}")
    # Chroma.from_documents does two things:
    # 1. Calls the embedding model to convert the text of each chunk into a vector.
    # 2. Stores the vector and the original text in the local directory.
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=VECTORSTORE_DIR
    )
    
    print("Successfully stored chunks in vector database.")
    return vectorstore

def get_retriever():
    """
    Initializes a retriever object from the existing ChromaDB.
    
    The Retriever is the interface that takes a query, converts it to an embedding,
    compares it against the database (using Cosine Similarity internally), and returns
    the top K most similar chunks.
    
    Returns:
        VectorStoreRetriever: The LangChain retriever object.
    """
    embeddings = get_embedding_model()
    
    # Load the existing database from disk
    vectorstore = Chroma(
        persist_directory=VECTORSTORE_DIR,
        embedding_function=embeddings
    )
    
    # Create a retriever that returns the top 3 most relevant chunks
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    return retriever
