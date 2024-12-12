import streamlit as st
import os
from rag.document_embedder import embedder_by_huggingface
from rag.vector_storage import load_vectorstorage
from langchain_components.custom_prompts import custom_chat_prompt
from langchain_components.inference_chain import inference_chain_with_prompt_template_rag
from langchain_components.llm_selection import llms_by_groq
from rag.vector_storage import create_vector_store_from_uploaded_file_streamlit

st.title("Ask anything about the paper 'Attention is all you need' or anything you upload.")
user_input = st.text_input("Enter your question here:")
uploaded_file = st.file_uploader("Upload a file", type=["pdf"])
embedder = embedder_by_huggingface(model="all-MiniLM-L12-v2")
vector_store_path = "./app/faiss_db_from_streamlit_file_huggingface"


if "button_clicked" not in st.session_state:
    st.session_state.button_clicked = False

create_vector_store_button = st.button("Create vector store from uploaded file")

# Update session state only when the button is clicked
if create_vector_store_button:
    st.session_state.button_clicked = True



def vector_store():
    if uploaded_file is None:
        if os.path.exists(vector_store_path):
            return load_vectorstorage(filename=vector_store_path,
                                    embedder=embedder,
                                    vectordbtype="faiss")
    elif st.session_state.button_clicked == True:
        if uploaded_file is not None:
            return create_vector_store_from_uploaded_file_streamlit(uploaded_file, embedder)

prompt=[
    ("system", "You are a helpful assistant who has to think step by step based on the provided context {context}"),
    ("user", "{question}"),
    ("system", "Answer the question only from the context provided and nothing else. If you don't know the answer, just say that you don't know.")
]
prompt_template = custom_chat_prompt(template=prompt)
llm = llms_by_groq(model="mixtral-8x7b-32768")

def chain():
    return inference_chain_with_prompt_template_rag(llm=llm,
                                                    vectorstorage=vector_store(),
                                                    prompt_template=prompt_template)

if user_input:
    st.write(chain().invoke({"query": user_input}))