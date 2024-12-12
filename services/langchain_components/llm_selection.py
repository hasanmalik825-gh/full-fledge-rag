from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq
from langchain_huggingface import ChatHuggingFace
from constants import GROQ_API_KEY


def llm_by_openai(model: str = "gpt-4o-mini", temperature: float = 0.5) -> ChatOpenAI:
    """
    This function is used to create the llm model using openai.
    """

    llm=ChatOpenAI(model=model, temperature=temperature)
    return llm

def llm_by_ollama(model: str = "llama3.2:1b", temperature: float = 0.5) -> ChatOllama:
    """
    This function is used to create the llm model using ollama.
    """

    llm=ChatOllama(model=model, temperature=temperature)
    return llm

def llm_by_huggingface(model: str = "BAAI/bge-small-en-v1.5", temperature: float = 0.5) -> ChatHuggingFace:
    """
    This function is used to create the llm model using huggingface.
    """
    llm=ChatHuggingFace(model=model, temperature=temperature)
    return llm

def llms_by_groq(model: str = "mixtral-8x7b-32768",
                temperature: float = 0.7) -> ChatGroq:
    """
    This function is used to create the llm model using groq.
    Models useable: ['gemma2-9b-it',
                    'llama3-groq-70b-8192-tool-use-preview',
                    'whisper-large-v3-turbo',
                    'llama-3.1-8b-instant',
                    'llama3-groq-8b-8192-tool-use-preview',
                    'llama-guard-3-8b',
                    'llama-3.2-90b-vision-preview',
                    'llama3-8b-8192',
                    'llama-3.2-11b-vision-preview',
                    'mixtral-8x7b-32768',
                    'llama-3.1-70b-versatile',
                    'llava-v1.5-7b-4096-preview',
                    'llama3-70b-8192',
                    'distil-whisper-large-v3-en',
                    'llama-3.2-1b-preview',
                    'llama-3.2-3b-preview',
                    'whisper-large-v3',
                    'gemma-7b-it']
    """

    llm=ChatGroq(model=model,
                 temperature=temperature,
                 groq_api_key=GROQ_API_KEY,
                 streaming=True)
    return llm
