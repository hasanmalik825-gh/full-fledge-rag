import os

OPENAI_API_KEY=os.environ.get("OPENAI_API_KEY")
OPENAI_MODEL_NAME = os.environ.get("OPENAI_MODEL_NAME") or "gpt-4o-mini"
## Langsmith Tracking
LANGCHAIN_API_KEY=os.environ.get("LANGCHAIN_API_KEY")
LANGCHAIN_TRACING_V2=os.environ.get("LANGCHAIN_TRACING_V2") or "true"
LANGCHAIN_PROJECT=os.environ.get("LANGCHAIN_PROJECT")
LANGCHAIN_ENDPOINT=os.environ.get("LANGCHAIN_ENDPOINT") or "https://api.smith.langchain.com"
GROQ_API_KEY=os.environ.get("GROQ_API_KEY")

IP_WHITELIST = os.environ.get("IP_WHITELIST") or ["127.0.0.1"]