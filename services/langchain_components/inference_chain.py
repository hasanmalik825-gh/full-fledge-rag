from langchain.chains import RetrievalQA
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory, SQLChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables.base import RunnableConfig
from langchain_core.runnables.base import Runnable
from typing import Optional, Any
from langchain_core.runnables.utils import Input, Output
from services.langchain_components.llm_selection import llms_by_groq
import sqlite3
from services.langchain_components.message_trimmer import trim_tokens_in_messages
from typing import Sequence
from langchain_core.messages import BaseMessage
import json
store={}

def _get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

def _insert_history_messages_sqlite(session_id: str, database_name: str, history_messages: json, message_type: str):
    history_messages_json = json.dumps({"type": message_type, "data": history_messages})
    conn = sqlite3.connect(database_name)
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE IF NOT EXISTS history_messages \
                   (id INTEGER PRIMARY KEY AUTOINCREMENT,session_id \
                   TEXT NOT NULL,message TEXT NOT NULL,UNIQUE(session_id, message))")
    conn.commit()
    try:
        cursor.execute("INSERT INTO history_messages (session_id, message) VALUES (?, ?)", (session_id, history_messages_json))
        conn.commit()
    except sqlite3.IntegrityError:
        pass
    conn.close()

def _insert_topic_name_sqlite(session_id: str, topic_name: str, database_name: str):
    conn = sqlite3.connect(database_name)
    cursor = conn.cursor()
    # create new table topics with session_id, topic_name where session_id is primary key
    cursor.execute("CREATE TABLE IF NOT EXISTS topics (session_id TEXT PRIMARY KEY, topic_name TEXT)")
    conn.commit()
    # insert new row into topics table with session_id and topic_name
    try:
        cursor.execute("INSERT INTO topics (session_id, topic_name) VALUES (?, ?)", (session_id, topic_name))
        conn.commit()
    except sqlite3.IntegrityError:
        pass
    conn.close()

def _create_runnable_with_history_input() -> Runnable:
    llm = llms_by_groq()
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant. Answer to the point and concise only."),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}"),
        ]
    )

    return  prompt | llm

def inference_chain_chat_history_sqlite(session_id: str,
                                        topic_name: str,
                                        database_name: str="sqlite.db",
                                        trim_messages: dict={"should_trim": False},
                                        **kwargs: Any) -> Runnable:
    """
    This function is used to create the inference chain with chat history in each session in sqlite database.
    args:
        session_id: session id
        topic_name: topic name
        database_name: database name

    default values: database_name="sqlite.db", llm=llms_by_groq(), input_messages_key="question", history_messages_key="history".

    default prompt: ("system", "You are a helpful assistant. Answer to the point and concise only."),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}"),

    kwargs are passed to trim_tokens_in_messages
    kwargs can be llm: ((list[BaseMessage]) -> int) | ((BaseMessage) -> int) | BaseLanguageModel,
    max_tokens: int = 1000,
    strategy: str = "last",
    include_system: bool = True,
    allow_partial: bool = False,
    start_on: str = "human"
    """
    def get_session_history_messages():
        past_messages = SQLChatMessageHistoryCustom(
        session_id=session_id, connection=f"sqlite:///{database_name}"
    )
        # If trimming is enabled, trim the messages and add in history table
        if trim_messages["should_trim"]:
            trimmed_messages = trim_tokens_in_messages(past_messages.messages, **kwargs)
            past_messages.add_messages(trimmed_messages, add_history_to_db=True, database_name=database_name)  # Add the trimmed messages
        # Return the processed message history
        return past_messages

    history_chain = RunnableWithMessageHistoryCustom(
        runnable=_create_runnable_with_history_input(),
        session_id=session_id,
        topic_name=topic_name,
        database_name=database_name,
        get_session_history=get_session_history_messages,
        input_messages_key="question",
        history_messages_key="history",
    )
    return history_chain

#inference with chat history in each session
def inference_chain_chat_history(session_id: str, trim_messages: dict={"should_trim": False}, **kwargs: Any) -> Runnable:
    """
    This function is used to create the inference chain with chat history in each session in memory(RAM).
    args:
        session_id: session id
        trim_messages: dictionary with key "should_trim" and value boolean default is False
        kwargs are passed to trim_messages kwargs can be

        llm: ((list[BaseMessage]) -> int) | ((BaseMessage) -> int) | BaseLanguageModel,
        max_tokens: int = 1000,
        strategy: str = "last",
        include_system: bool = True,
        allow_partial: bool = False,
        start_on: str = "human"

    default values: llm=llms_by_groq(), input_messages_key="question", history_messages_key="history".

    default prompt: ("system", "You are a helpful assistant. Answer to the point and concise only."),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}"),
    """
    def get_session_history_messages():
        past_messages=_get_session_history(session_id=session_id)
        if trim_messages["should_trim"]:
            past_messages.messages=trim_tokens_in_messages(past_messages.messages, **kwargs)
        return past_messages

    history_chain = RunnableWithMessageHistory(
    runnable=_create_runnable_with_history_input(),
    get_session_history=get_session_history_messages,
    input_messages_key="question",
    history_messages_key="history",
    )
    return history_chain

def inference_chain(llm, prompt_template, output_parser=None) -> Runnable:
    """
    This function is used to create the inference chain with a custom prompt.
    """
    chain = prompt_template | llm
    if output_parser:
        chain = chain | output_parser
    return chain

def inference_chain_rag(
        llm, 
        vectorstorage, 
        prompt_template, 
        output_parser=None, 
        return_source_documents=False, 
        k=3
    ) -> Runnable:
    """
    This function is used to create RetrievalQ with "stuff" type and also take retriever for rag.
    args:
        llm: langchain llm
        vectorstorage: vector store
        prompt_template: prompt template
        output_parser: output parser
        return_source_documents: return source documents
        k: number of documents to return
    """

    # Create a retriever from the vector store
    retriever = vectorstorage.as_retriever(search_kwargs={'k': k})
    
    # Create the chain with the specified parameters
    # setting config to return source documents
    chain = RetrievalQA.from_chain_type(
        llm=llm, 
        chain_type="stuff", 
        retriever=retriever, 
        chain_type_kwargs={"prompt": prompt_template}
    )
    chain.return_source_documents = return_source_documents
    if output_parser:
        chain = chain | output_parser
    return chain


class RunnableWithMessageHistoryCustom(RunnableWithMessageHistory):
    """
    Custom RunnableWithMessageHistory implementation that stores session_id and topic_name in database when invoked.
    """
    session_id: str
    topic_name: str
    database_name: str

    def invoke(
        self,
        input: Input,
        config: Optional[RunnableConfig] = None,
        **kwargs: Optional[Any],
    ) -> Output:
        # Insert topic name before invoking
        _insert_topic_name_sqlite(self.session_id, self.topic_name, self.database_name)
        
        return self.bound.invoke(
            input,
            self._merge_configs(config),
            **{**self.kwargs, **kwargs},
        )
    
class SQLChatMessageHistoryCustom(SQLChatMessageHistory):

    def add_messages(
        self,
        messages: Sequence[BaseMessage],
        add_history_to_db: Optional[bool]=False,
        database_name: Optional[str]=None
    ) -> None:
        # Add all messages in one transaction
        with self._make_sync_session() as session:
            if add_history_to_db:
                for message in self.messages:
                    _insert_history_messages_sqlite(self.session_id, database_name, message.model_dump(), message.type)
                self.clear()
            for message in messages:
                session.add(self.converter.to_sql_model(message, self.session_id))
            session.commit()
