from langchain_core.prompts import PromptTemplate, ChatPromptTemplate

def custom_prompt(template: str) -> PromptTemplate:
    """
    "This function is used to create a custom prompt.
    example: template = "You are a helpful assistant. You are given the following pieces of retrieved context: {context}. \
    The user asked the following question: {question}"
    """
    
    prompt = PromptTemplate(template=template)
    return prompt

def custom_chat_prompt(template: list[tuple[str, str]]) -> ChatPromptTemplate:
    """
    This function is used to create a custom chat prompt.
    You can only use one of 'human', 'user', 'ai', 'assistant', or 'system' as role.
    example: template = [
        ("system", "You are a helpful assistant. You are given the following pieces of retrieved context: {context}."),
        ("human", "Who is leonardo di caprio's gf?"),
        ("ai", "Leonardo DiCaprio's girlfriend is Gisele Bündchen. They were married from 2000 to 2016.")
    ]
    """
    prompt = ChatPromptTemplate(messages=template)
    return prompt
