from flashrank import Ranker, RerankRequest
from typing import List, Dict, Any, Union
from langchain.schema import Document
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import FlashrankRerank
from langchain_community.vectorstores import VectorStore
from langchain.chains import RetrievalQA
from langchain_core.language_models import BaseChatModel

def document_flash_reranker(
        query:str,
        documents:List[Dict[str, Any]],
        model_name:str="ms-marco-MiniLM-L-12-v2",
        cache_dir:str="/rerankerchache"
    )->List[Dict[str, Any]]:
    """
    Reranks a list of documents based on a query using the FlashRank model.
    Args:
        query: The query to rerank the documents.
        documents: The list of documents to rerank.
        model_name: The name of the model to use for reranking. model_name can be one of the following:
            - ms-marco-TinyBERT-L-2-v2
            - ms-marco-MiniLM-L-12-v2
            - rank-T5-flan
            - ms-marco-MultiBERT-L-12
            - ms-marco-MiniLM-L-6-v2
            - rank_zephyr_7b_v1_full (Large 4GB)
        cache_dir: The directory to store the model cache.
    Returns:
        The reranked list of documents.
    """
    if documents[0].metadata or documents[0].page_content:
        #change name of page_content to text and metadata to meta
        documents = [{'text':i.page_content, 'meta':i.metadata} for i in documents]
        #print(documents[0])
    ranker = Ranker(model_name=model_name, cache_dir=cache_dir) # Small (~34MB), slightly slower & best performance (ranking precision).
    rerank_request = RerankRequest(query=query, passages=documents)
    rerank_response = ranker.rerank(rerank_request)
    return rerank_response

def document_flash_reranker_langchain(
        query:str,
        retriever:VectorStore,
        return_llm_response:bool=False,
        llm:BaseChatModel=None
    )->Union[List[Document], BaseChatModel]:
    """
    Returns the llm response based on the reranked documents if return_llm_response is True otherwise returns the reranked documents.
    Args:
        query: The query to rerank the documents.
        retriever: The retriever to use for reranking.
        return_llm_response: Whether to return the response from the llm.
        llm: The llm to use for inferencing on reranked documents.
    Returns:
        The reranked list of documents or the llm response.
    """
    compressor = FlashrankRerank()
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=retriever
    )
    if return_llm_response:
        return RetrievalQA.from_chain_type(llm=llm, retriever=compression_retriever).invoke(query)
    else:
        return compression_retriever.invoke(query)







