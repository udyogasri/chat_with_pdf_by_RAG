import logging
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from rag.retriever import get_retriever
from config import OLLAMA_MODEL, OLLAMA_BASE_URL

logger = logging.getLogger(__name__)

def get_rag_chain():
    """
    Constructs the RAG pipeline with Conversation History support.
    
    This function uses LangChain's built-in chains which handle the complex
    routing of passing the chat history to contextualize the question, retrieving
    documents, and then passing the documents and history to the LLM to get the final answer.
    
    Returns:
        Chain: A LangChain retrieval chain, or None if failed.
    """
    try:
        logger.info(f"Initializing Ollama LLM: {OLLAMA_MODEL}")
        llm = OllamaLLM(
            model=OLLAMA_MODEL,
            base_url=OLLAMA_BASE_URL
        )
        
        retriever = get_retriever()
        if not retriever:
            logger.error("Failed to initialize retriever. Cannot build RAG chain.")
            return None

        # 1. History-Aware Retriever Prompt
        # This asks the LLM to rewrite the user's question to be standalone,
        # using the context from the chat history.
        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        history_aware_retriever = create_history_aware_retriever(
            llm, retriever, contextualize_q_prompt
        )
        
        # 2. Document QA Prompt
        # This tells the LLM how to answer the question using the retrieved context.
        qa_system_prompt = (
            "You are an expert HR assistant and resume analyzer.\n"
            "You will be provided with excerpts from multiple resumes below. Each excerpt is labeled with its 'Source File'.\n\n"
            "INSTRUCTIONS:\n"
            "- If the user asks for a candidate matching specific criteria, evaluate the excerpts and select the SINGLE best matching candidate.\n"
            "- Provide details for ONLY that specific candidate. Do NOT mix information from different resumes.\n"
            "- If you find a match, you MUST include the exact 'Source File' name in your response so the user knows which resume it is.\n"
            "- If no candidate matches, clearly state that you couldn't find a match.\n\n"
            "CONTEXT:\n{context}"
        )
        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", qa_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        
        # We customize how each document is formatted so the LLM knows the source file name
        from langchain_core.prompts import PromptTemplate
        document_prompt = PromptTemplate.from_template("Source File: {source}\nContent:\n{page_content}")
        
        # The 'stuff' documents chain takes all the retrieved documents and stuffs them into the {context} variable
        question_answer_chain = create_stuff_documents_chain(
            llm=llm, 
            prompt=qa_prompt, 
            document_prompt=document_prompt
        )
        
        # 3. Complete Retrieval Chain
        # This brings it all together: it takes the input, uses the history-aware retriever to get documents,
        # and then uses the question_answer_chain to generate the final answer.
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
        
        logger.info("Successfully built history-aware RAG chain.")
        return rag_chain
    except Exception as e:
        logger.error(f"Error building RAG chain: {e}")
        return None
