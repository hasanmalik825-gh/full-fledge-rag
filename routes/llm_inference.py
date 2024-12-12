from fastapi import APIRouter
from services.langchain_components.inference_chain import inference_chain
from services.langchain_components.llm_selection import llms_by_groq
from services.langchain_components.custom_prompts import custom_prompt
from langchain_core.output_parsers import StrOutputParser

llm_inference_router = APIRouter()

@llm_inference_router.post("/llm-inference")
def llm_inference(query: str):
    chain = inference_chain(
        llm=llms_by_groq(),
        prompt_template=custom_prompt("you are physics teacher explain complex concepts in simpler way"),
        output_parser=StrOutputParser()
    )
    return chain.invoke({"query": query})