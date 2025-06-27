from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.output_parsers import StrOutputParser
from tools.react_prompt_template import get_prompt_template
from tools.pdf_query_tools import indian_constitution_pdf_query, indian_laws_pdf_query
import warnings
import os


def agent(query: str, use_ollama: bool = False, ollama_model: str = "llama3.1:8b"):
    """
    Create and run an agent with either Groq or OLLAMA LLM
    
    Args:
        query (str): The user's query
        use_ollama (bool): Whether to use OLLAMA instead of Groq
        ollama_model (str): OLLAMA model to use (default: llama3.1:8b)
    """
    warnings.filterwarnings("ignore", category=FutureWarning)

    # Set environment variables for PDF tools to use the same LLM choice
    if use_ollama:
        os.environ["USE_OLLAMA"] = "true"
        os.environ["OLLAMA_MODEL"] = ollama_model
        # Support Docker OLLAMA or local OLLAMA
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        # Use OLLAMA for local LLM
        LLM = ChatOllama(
            model=ollama_model,
            base_url=base_url,
            temperature=0.1,
            timeout=120,  # Increase timeout for local models
        )
        print(f"Using OLLAMA model: {ollama_model} at {base_url}")
    else:
        os.environ["USE_OLLAMA"] = "false"
        # Use Groq for cloud LLM
        LLM = ChatGroq(model="llama3-8b-8192")
        print("Using Groq model: llama3-8b-8192")

    tools = [indian_constitution_pdf_query, indian_laws_pdf_query]
    prompt_template = get_prompt_template()

    agent = create_react_agent(
        LLM,
        tools,
        prompt_template
    )

    agent_executor = AgentExecutor(
        agent=agent, 
        tools=tools, 
        verbose=False, 
        handle_parsing_errors=True,
        max_iterations=10,  # Limit iterations for local models
        early_stopping_method="generate"
    )

    try:
        result = agent_executor.invoke({"input": query})
        return result["output"]
    except Exception as e:
        error_msg = f"Error processing query: {str(e)}"
        print(error_msg)
        return f"Sorry, I encountered an error while processing your query: {str(e)}"


def get_available_ollama_models():
    """
    Get list of available OLLAMA models
    """
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            return [model["name"] for model in models]
        else:
            return []
    except Exception as e:
        print(f"Could not connect to OLLAMA: {e}")
        return []


def check_ollama_connection():
    """
    Check if OLLAMA is running and accessible
    """
    try:
        import requests
        response = requests.get("http://localhost:11434/api/version", timeout=5)
        return response.status_code == 200
    except:
        return False