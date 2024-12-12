import uvicorn
# create langserve endpoint
from langserve import add_routes
from fastapi import FastAPI
from services.langchain_components.inference_chain import inference_chain
from services.langchain_components.llm_selection import llms_by_groq
from services.langchain_components.custom_prompts import custom_chat_prompt
from langchain_core.output_parsers import StrOutputParser

app = FastAPI(title="RAG API",
              description="API for RAG",
              version="0.0.1")

prompt = [('system', 'you are {subject} teacher explain complex concepts in simpler way'),
          ('human', 'explain the {concept} concisely')]
chain = inference_chain(
    llm=llms_by_groq(),
    prompt_template=custom_chat_prompt(prompt),
    output_parser=StrOutputParser()
)

# Add middleware
@app.middleware("http")
async def custom_middleware(request, call_next):
    # Pre-processing (before request is handled)
    print("Middleware: Before request")
    response = await call_next(request)
    # Post-processing (after response is generated)
    print("Middleware: After request")
    return response

add_routes(app, runnable=chain, path="/rag")

if __name__ == "__main__":
    uvicorn.run(app=app, host="localhost", port=8000, reload=True)

    