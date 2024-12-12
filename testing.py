#from utils.chat_history import get_chat_history
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from services.langchain_components.llm_selection import llms_by_groq
from services.langchain_components.custom_prompts import custom_chat_prompt
from langchain_core.messages import AIMessage, HumanMessage
from services.langchain_components.message_trimmer import trim_tokens_in_messages
from typing import Any


chat_prompt = custom_chat_prompt(template=[
    ("ai", 'You are a helpful assistant'),
    ("human",'My name is Hasan')
])
llm=llms_by_groq()
store={}

def get_session_history(session_id: str, trim_messages: bool= False, **kwargs: Any) -> BaseChatMessageHistory:
    """
    kwargs are passed to trim_tokens_in_messages
    kwargs can be llm: ((list[BaseMessage]) -> int) | ((BaseMessage) -> int) | BaseLanguageModel,
    max_tokens: int = 1000,
    strategy: str = "last",
    include_system: bool = True,
    allow_partial: bool = False,
    start_on: str = "human"
    """
    if session_id not in store:
        store[session_id] = ChatMessageHistory()

    # Trim messages in place
    if trim_messages:
        trimmed = trim_tokens_in_messages(messages=store[session_id].messages,
                        **kwargs
                        )
        store[session_id].messages = trimmed  # Update the messages directly

    return store[session_id]


message_history = RunnableWithMessageHistory(llm, get_session_history)