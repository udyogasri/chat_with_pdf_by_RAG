import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import CHUNK_SIZE, CHUNK_OVERLAP

def load_and_split_pdf(file_path: str):
    """
    Loads a PDF file and splits it into chunks.
    
    Why chunking?
    LLMs have a context window limit (they can only read a certain number of tokens at once).
    Also, searching through a massive document is inefficient. By breaking a document into
    smaller "chunks", we can search for the most relevant pieces of information and only send
    those specific pieces to the LLM.

    Args:
        file_path (str): The absolute path to the uploaded PDF file.
        
    Returns:
        list: A list of Document objects containing the chunked text and metadata.
    """
    print(f"Loading PDF from: {file_path}")
    
    # 1. Load the PDF
    # PyPDFLoader reads the PDF and converts each page into a LangChain 'Document' object.
    loader = PyPDFLoader(file_path)
    pages = loader.load()
    
    print(f"Loaded {len(pages)} pages. Starting chunking process...")
    
    # 2. Initialize the Text Splitter
    # RecursiveCharacterTextSplitter is the recommended generic text splitter.
    # It tries to split on paragraphs (\n\n), then sentences (\n), then spaces, etc.,
    # keeping related text together as much as possible.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,       # Maximum number of characters in a chunk
        chunk_overlap=CHUNK_OVERLAP, # Number of characters to overlap between chunks
        length_function=len          # Function used to measure the length
    )
    
    # 3. Split the loaded pages into chunks
    chunks = text_splitter.split_documents(pages)
    
    print(f"Created {len(chunks)} chunks.")
    return chunks
