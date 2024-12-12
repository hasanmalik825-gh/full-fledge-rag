from pydantic import BaseModel, Field
from langchain.output_parsers import PydanticToolsParser
from langchain_core.prompts import ChatPromptTemplate

class QueryReformulation(BaseModel):
    """You have performed query reformulation to generate a paraphrasing of a query."""

    reformulated_query: str = Field(
        ...,
        description="A unique paraphrasing of the original question.",
    )

def query_expansion(question: str, llm_model: BaseModel) -> str:
    system = """You are an expert at converting user questions into refined and expanded queries for retrieving \
relevant information from a document database.

Return at least 3 distinct paraphrased versions of the original question.

Perform query expansion by generating multiple phrasings of the user's question. Ensure the expanded queries:

* Include common synonyms for key terms in the question.
* Represent different ways the question might commonly be phrased.
* Maintain the original meaning and intent of the question.
* If there are acronyms or words that you are not familiar with, do not attempt to rephrase them.
"""
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "{question}"),
        ]
    )
    llm = llm_model
    llm_with_tools = llm.bind_tools([QueryReformulation])
    query_analyzer = prompt | llm_with_tools | PydanticToolsParser(tools=[QueryReformulation])
    query_analyzer_result=query_analyzer.invoke({"question": question})
    print(query_analyzer_result)
    return [query.reformulated_query for query in query_analyzer_result]

def query_rewriting(question: str, llm_model: BaseModel) -> str:
    """Refines and clarifies the user's question for better processing or retrieval."""
    system = """You are an expert at rewriting user questions to improve clarity and focus.

Return a single refined version of the original question. The rewritten query should:

* Use simpler, more precise language without changing the intent.
* Remove ambiguity or redundancy in the original question.
* Ensure the rewritten question is more concise and focused.
"""
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "{question}"),
        ]
    )
    llm = llm_model
    llm_with_tools = llm.bind_tools([QueryReformulation])
    query_analyzer = prompt | llm_with_tools | PydanticToolsParser(tools=[QueryReformulation])
    query_analyzer_result = query_analyzer.invoke({"question": question})
    print(query_analyzer_result)
    return [query.reformulated_query for query in query_analyzer_result]

def query_decomposition(question: str, llm_model: BaseModel) -> list[str]:
    """Breaks down a complex question into smaller, specific sub-queries."""
    system = """You are an expert at breaking down complex questions into smaller, \
focused sub-queries or separating distinct questions when multiple are asked.

Decompose the user's input as follows:

* If multiple distinct questions are provided, return them as separate queries without modification.
* Ensure each distinct question is precise, and can be independently processed or \
retrieved from a document database.
* Together, the distinct questions should fully address the intent of the original input.
"""
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "{question}"),
        ]
    )
    llm = llm_model
    llm_with_tools = llm.bind_tools([QueryReformulation])
    query_analyzer = prompt | llm_with_tools | PydanticToolsParser(tools=[QueryReformulation])
    query_analyzer_result = query_analyzer.invoke({"question": question})
    print(query_analyzer_result)
    return [query.reformulated_query for query in query_analyzer_result]

