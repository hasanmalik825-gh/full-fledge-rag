from langchain_core.messages import trim_messages
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import BaseMessage
from langchain_core.prompt_values import PromptValue
from typing import Any, Callable, Iterable, Union

MessageLikeRepresentation = Union[BaseMessage, list[str], tuple[str, str], str, dict[str, Any]]



def trim_tokens_in_messages(
        messages: Union[Iterable[MessageLikeRepresentation], PromptValue],
        llm: Union[
        Callable[[list[BaseMessage]], int],
        Callable[[BaseMessage], int],
        BaseLanguageModel],
        max_tokens: int = 1000,
        strategy: str = "last",
        include_system: bool = True,
        allow_partial: bool = False,
        start_on: str = "human"
    ) -> list[BaseMessage]:
    """
    Trims the messages to the given max_tokens using the specified strategy.
    Args:
        messages: list[BaseMessage]
        llm: model to use for token counting
        max_tokens: int maximum number of tokens to keep
        strategy: "last" or "first" whether to trim from the end or the beginning
        include_system: bool include system messages in the trimming
        allow_partial: bool allow partial trimming
        start_on: "human" or "ai" where to start trimming from
    """
    trimmer=trim_messages(
        max_tokens=max_tokens,
        strategy=strategy,
        token_counter=llm,
        include_system=include_system,
        allow_partial=allow_partial,
        start_on=start_on
    )
    return trimmer.invoke(messages)