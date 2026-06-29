from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from rag.retriever import get_retriever
from config import OLLAMA_MODEL, OLLAMA_BASE_URL

def get_rag_chain():
    """
    Connects the components: Retriever -> Prompt -> LLM -> Output
    
    This uses LangChain's Expression Language (LCEL) to create a data pipeline.
    
    What happens internally:
    1. The user provides a question.
    2. 'RunnablePassthrough' passes the question to the Retriever.
    3. The Retriever fetches relevant chunks from ChromaDB (this becomes the 'context').
    4. The 'context' and 'question' are injected into the PromptTemplate.
    5. The formatted prompt is sent to the local Ollama LLM.
    6. StrOutputParser cleans up the LLM's raw output into a simple string.
    
    Returns:
        Runnable: The complete RAG chain ready to be invoked.
    """
    
    # 1. Initialize the local LLM via Ollama
    llm = OllamaLLM(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL
    )
    
    # 2. Get the Retriever we built in retriever.py
    retriever = get_retriever()
    
    # 3. Define the Prompt Template
    # We tell the LLM exactly how to behave. We provide it with the extracted context
    # and strictly tell it not to guess if it doesn't know the answer.
    template = """
    You are a helpful and knowledgeable AI assistant.
    Use the following pieces of retrieved context to answer the question. 
    If you don't know the answer, just say that you don't know. 
    Do not make up an answer. Keep the answer clear and concise.

    Context:
    {context}

    Question: 
    {question}

    Answer:
    """
    
    prompt = PromptTemplate.from_template(template)
    
    # Function to format the documents returned by the retriever into a single string
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    # 4. Construct the LangChain Chain (LCEL)
    # The '|' operator is used by LangChain to pipe the output of one step into the next.
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain
