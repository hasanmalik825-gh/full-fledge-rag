from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores import Chroma
from langchain_core.vectorstores import VectorStore
from langchain_core.documents import Document
from constants import *
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores.utils import filter_complex_metadata
from services.rag.document_loader import pdf_loader
from services.rag.document_chunker import recursive_character_splitter
from services.rag.document_embedder import embedding_maker
from io import BytesIO
import os
from utils.file_conversion import bytes_to_pdf

def create_faiss_db(documents=None, embedder=None, embeddings=None, from_embeddings=False) -> FAISS:
    """
    This function is used to store the documents in the faiss database.
    """
    documents=filter_complex_metadata(documents)
    if from_embeddings:
        dummy_embeddings = DummyEmbeddings(embedder, embeddings=embeddings)
        vectorstoredb=FAISS.from_documents(documents, dummy_embeddings)
    else:
        vectorstoredb=FAISS.from_documents(documents, embedder)
    return vectorstoredb

def create_chroma_db(documents=None, embedder=None, embeddings=None, from_embeddings=False) -> Chroma:
    """
    This function is used to store the documents in the chroma database.
    """
    documents=filter_complex_metadata(documents)
    persist_directory='./chroma_db_'#+uuid.uuid1().__str__()

    if from_embeddings:
        dummy_embeddings = DummyEmbeddings(embedder, embeddings=embeddings)
        vectorstoredb=Chroma.from_documents(documents=documents,
                                            embedding=dummy_embeddings,
                                            persist_directory=persist_directory)
    else:
        vectorstoredb=Chroma.from_documents(documents=documents,
                                            embedding=embedder,
                                            persist_directory=persist_directory)
    return vectorstoredb

def generic_similerty_search(query, vectorstorage, k=3) -> list[Document]:
    """
    This function is used to search the similar documents to the query and return the top 3 results.
    """
    return vectorstorage.similarity_search(query=query, k=k)

def save_vectorstorage(vectorstorage, filename) -> None:
    """
    This function is used to save the vectorstorage to a file.
    """

    if type(vectorstorage).__name__=="FAISS":
        vectorstorage.save_local(filename)
    elif type(vectorstorage).__name__=="Chroma":
        vectorstorage.persist()

def load_vectorstorage(
    filename:str,
    embedder:Embeddings,
    vectordbtype:str
) -> VectorStore:
    """
    This function is used to load the vectorstorage from a file.
    """
    if vectordbtype.lower()=="faiss":
        vectorstorage=FAISS.load_local(
            folder_path=filename,
            embeddings=embedder,
            allow_dangerous_deserialization=True
        )
    elif vectordbtype.lower()=="chroma":
        vectorstorage=Chroma(
            persist_directory=filename,
            embedding_function=embedder
        )
    return vectorstorage

def create_vector_store_from_uploaded_file_streamlit(
    uploaded_file:BytesIO,
    embedder:Embeddings, 
    vectordbtype:str = "faiss"
) -> FAISS:
    """
    This function is used to create the vector store from the uploaded file (which currently is only pdf).
    """
    temp_file_path = bytes_to_pdf(uploaded_file.getvalue())
    documents = pdf_loader(temp_file_path)
    os.unlink(temp_file_path)

    documents = recursive_character_splitter(
        document=documents,
        chunk_size=500,
        chunk_overlap=100
    )
    embeddings = embedding_maker(embedder=embedder, documents=documents)
    if vectordbtype.lower()=="faiss":
        vector_store = create_faiss_db(
            documents=documents,
            embedder=embedder,
            embeddings=embeddings,
            from_embeddings=True
        )
        vector_store.save_local("app/faiss_db_from_streamlit_file_huggingface")
    elif vectordbtype.lower()=="chroma":
        vector_store = create_chroma_db(
            documents=documents,
            embedder=embedder,
            embeddings=embeddings,
            from_embeddings=True
        )
        vector_store.persist()
    return vector_store

class DummyEmbeddings(Embeddings):
    """
    This class is used to create a dummy embeddings object.
    """
    def __init__(self, embedder, embeddings=None):
        self.embedder = embedder
        self.embeddings = embeddings

    def embed_query(self, text: str) -> list[float]:
        return self.embedder.embed_query(text)
    
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embeddings