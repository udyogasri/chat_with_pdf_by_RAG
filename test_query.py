import os
import sys
import logging

# Disable ChromaDB telemetry properly
os.environ["CHROMA_TELEMETRY_DISABLED"] = "1"
os.environ["ANONYMIZED_TELEMETRY"] = "False"

# Configure logging
logging.basicConfig(level=logging.INFO)

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from rag.chain import get_rag_chain

if __name__ == "__main__":
    chain = get_rag_chain()
    if not chain:
        print("Failed to get chain")
        sys.exit(1)
        
    print("Testing chain with input: 'What is the name of the candidate?'")
    try:
        res = chain.invoke({
            "input": "What is the name of the candidate?",
            "chat_history": []
        })
        print(f"Answer: {res['answer']}")
        print(f"Context docs: {len(res['context'])}")
    except Exception as e:
        print(f"Exception during invoke: {e}")
