from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_text_splitters import RecursiveJsonSplitter
from langchain_core.documents import Document
from constants import *

def recursive_character_splitter(document : list[Document], chunk_size : int, chunk_overlap : int) -> list[Document]:
    """
    This function is used to split the documents into chunks using recursive character text splitter.
    Args:
        document: list[Document]: The document to be split.
        chunk_size: int: The size of the chunk to be split.
        chunk_overlap: int: The overlap of the chunk to be split.
    Returns:
        List[Document]: A list of documents.
    """

    splitter=RecursiveCharacterTextSplitter(chunk_size=chunk_size,chunk_overlap=chunk_overlap)
    documents=splitter.split_documents(document)
    return documents

def recursive_json_splitter(json_doc : dict, chunk_size : int) -> list[Document]:
    """
    This function is used to split the json documents into chunks using recursive json text splitter.
    Args:
        json_doc: dict: The json document to be split.
        chunk_size: int: The size of the chunk to be split.
    Returns:
        List[Document]: A list of documents.
    """

    splitter=RecursiveJsonSplitter(max_chunk_size=chunk_size)
    documents=splitter.split_text(json_doc)
    return documents

