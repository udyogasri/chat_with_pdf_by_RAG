from langchain_huggingface import HuggingFaceEmbeddings
from config import EMBEDDING_MODEL_NAME

def get_embedding_model():
    """
    Initializes and returns the HuggingFace embedding model.
    
    Why Embeddings?
    Computers don't understand text; they understand numbers.
    An embedding model reads text and converts it into a dense vector (a long array of numbers).
    Words or sentences with similar meanings will have vectors that are closer to each other in 
    the mathematical space.
    
    We use BAAI/bge-small-en-v1.5 because it is a highly capable, small, and fast open-source
    embedding model, perfect for running locally.
    
    Returns:
        HuggingFaceEmbeddings: The initialized embedding model object.
    """
    print(f"Initializing embedding model: {EMBEDDING_MODEL_NAME}")
    
    # Initialize the HuggingFace embeddings
    # This downloads the model from HuggingFace (on first run) and loads it into memory.
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        # model_kwargs={'device': 'cpu'} # Uncomment if you explicitly want to run on CPU, otherwise it auto-detects GPU/CPU
    )
    
    return embeddings
