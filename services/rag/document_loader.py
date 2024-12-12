## Data Ingestion--From the website we need to scrape the data
from langchain_community.document_loaders import WebBaseLoader
import bs4
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import ArxivLoader
from langchain_community.document_loaders import WikipediaLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from constants import *

def text_loader(file) -> list[Document]:
    """
    This function is used to load the text files and extract the text from the file
    It returns a list of documents
    """

    docs=TextLoader(file).load()
    return docs

def web_loader(link) -> list[Document]:
    """
    This function is used to load the web pages and extract the text from the html tags
    It returns a list of documents
    """

    docs = WebBaseLoader(
    web_path=link,
    bs_kwargs={"parse_only": bs4.SoupStrainer(["h1", "h2", "h3", "p", "li", "a"])}
)
    return docs.load()

def arxiv_loader(query, only_summary=False) -> list[Document]:
    """
    This function is used to load the arxiv papers and extract the text from the arxiv api.
    It returns a list of documents
    It supports all arguments of `ArxivAPIWrapper`
    example: query = "reasoning", "1706.03762"
    """

    docs = ArxivLoader(
        query=query,
        load_max_docs=2,
        headers={"User-Agent": LANGCHAIN_PROJECT},
        # doc_content_chars_max=1000,
        # load_all_available_meta=False,
        # ...
    )
    if only_summary:
        return docs.get_summaries_as_docs()
    return docs.load()

def wiki_loader(query) -> list[Document]:
    """
    This function is used to load the wikipedia pages and extract the text from the wikipedia api.
    It returns a list of documents
    """

    docs = WikipediaLoader(query=query, load_max_docs=2).load()
    return docs

def pdf_loader(file) -> list[Document]:
    """
    This function is used to load the pdf files and extract the text from the pdf file.
    It returns a list of documents
    """
    
    docs=PyPDFLoader(file).load()
    return docs
