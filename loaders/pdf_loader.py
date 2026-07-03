import os
import logging
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import CHUNK_SIZE, CHUNK_OVERLAP

logger = logging.getLogger(__name__)

def load_and_split_pdf(file_path: str):
    """
    Loads a PDF file and splits it into chunks.
    
    Why chunking?
    LLMs have a context window limit. By breaking a document into
    smaller "chunks", we can search for the most relevant pieces of information 
    and only send those specific pieces to the LLM.

    Args:
        file_path (str): The absolute path to the uploaded PDF file.
        
    Returns:
        list: A list of Document objects containing the chunked text and metadata,
              or an empty list if an error occurs.
    """
    logger.info(f"Loading PDF from: {file_path}")
    
    try:
        # 1. Load the PDF
        # PyPDFLoader automatically injects metadata like 'source' and 'page' into each Document!
        loader = PyPDFLoader(file_path)
        pages = loader.load()
        
        logger.info(f"Loaded {len(pages)} pages from {os.path.basename(file_path)}. Starting chunking process...")
        
        # 2. Initialize the Text Splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len
        )
        
        # 3. Split the loaded pages into chunks
        chunks = text_splitter.split_documents(pages)
        
        logger.info(f"Created {len(chunks)} chunks for {os.path.basename(file_path)}.")
        return chunks
        
    except Exception as e:
        logger.error(f"Error loading or splitting PDF {file_path}: {e}")
        return []
