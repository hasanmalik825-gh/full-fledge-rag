from langchain_community.tools import ArxivQueryRun,WikipediaQueryRun, TavilySearchResults
from langchain_community.utilities import WikipediaAPIWrapper,ArxivAPIWrapper
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
from langchain_core.language_models import BaseChatModel
from langchain.agents import initialize_agent,AgentType,AgentExecutor
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.sql_database import SQLDatabase
from langchain import hub

def create_agent_executor(llm:BaseChatModel)->AgentExecutor:
    """
    Create an agent executor with the given LLM.
    Args:
        llm: The language model to use for the agent.
    Returns:
        An AgentExecutor instance.
    Default tools:
        ArxivQueryRun: Search for papers on Arxiv.
        WikipediaQueryRun: Search for information on Wikipedia.
        TavilySearchResults: Search for information on Tavily.

    Default prompt:
        "hwchase17/react" : 
 "Answer the following questions as best you can. You have access to the following tools:\
\n\n{tools}\n\nUse the following format:\n\nQuestion: the input question you must answer\nThought: you should always \
think about what to do\nAction: the action to take, should be one of [{tool_names}]\nAction Input: the input to the \
action\nObservation: the result of the action\n... (this Thought/Action/Action Input/Observation can repeat N \
times)\nThought: I now know the final answer\nFinal Answer: the final answer to the original input \
question\n\nBegin!\n\nQuestion: {input}\nThought:{agent_scratchpad}"
    """
    tools = [
        ArxivQueryRun(api_wrapper=ArxivAPIWrapper(
            top_k_results=1,
            doc_content_chars_max=500)
        ),
        WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(
            top_k_results=1,
            doc_content_chars_max=500)
        ),
        TavilySearchResults(
            api_wrapper=TavilySearchAPIWrapper(
            )
        )
    ]
    prompt = hub.pull("hwchase17/react")
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        handling_parsing_errors=True,
        agent_kwargs={"prompt":prompt}
        )
    return agent

def create_sql_agent_executor(llm:BaseChatModel=None,db:SQLDatabase=None)->AgentExecutor:
    agent = create_sql_agent(
        llm=llm,
        toolkit=SQLDatabaseToolkit(db=db,llm=llm),
        verbose=True,
        handling_parsing_errors=True,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION
    )
    return agent