from langchain_openai import OpenAIEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from constants import *
from langchain_core.embeddings import Embeddings

def embedding_maker(embedder:Embeddings, documents:list[Document]):
    """
    This function is used to embed the documents using openai embedding model.
    Args:
        model: str: The model to be used.
        dimensions: int: The dimensions of the embedding.
    Returns:
        Embeddings: The embeddings of the documents.
    """
    documents = [doc.page_content for doc in documents]
    embeddings=embedder.embed_documents(documents)
    return embeddings
    

def embedder_by_openai(model : str = "text-embedding-3-large", dimensions : int = 1536) -> OpenAIEmbeddings:
    """
    This function is used to embed the documents using openai embedding model.
    Args:
        model: str: The model to be used.
        dimensions: int: The dimensions of the embedding.
    Returns:
        OpenAIEmbeddings: The embedding model.
    """
    embedder = OpenAIEmbeddings(model=model, dimensions=dimensions)
    return embedder

def embedder_by_ollama(model : str = "llama3.2:1b") -> OllamaEmbeddings:
    """
    This function is used to embed the documents using ollama embedding model.
    Args:
        model: str: The model to be used.
    Returns:
        OllamaEmbeddings: The embedding model.
    """
    embedder = OllamaEmbeddings(
    model=model,
)
    return embedder

def embedder_by_huggingface(model : str = "all-MiniLM-L12-v2") -> HuggingFaceEmbeddings:
    """
    This function is used to embed the documents using huggingface embedding model.
    """
    embedder = HuggingFaceEmbeddings(model_name=model)
    return embedder
