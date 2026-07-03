import os

# Disable ChromaDB telemetry properly to prevent console errors
os.environ["CHROMA_TELEMETRY_DISABLED"] = "1"
os.environ["ANONYMIZED_TELEMETRY"] = "False"
# Base paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
VECTORSTORE_DIR = os.path.join(BASE_DIR, "vectorstore")

# Make sure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(VECTORSTORE_DIR, exist_ok=True)

# Chunking configurations (Smaller chunks = much higher accuracy for specific names)
CHUNK_SIZE = 300
CHUNK_OVERLAP = 50

# Embedding model configurations
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"

# LLM Configurations
# Assumes Ollama is running locally. You can use 'llama3.1', 'mistral', etc.
OLLAMA_MODEL = "llama3.1"
OLLAMA_BASE_URL = "http://localhost:11434"
