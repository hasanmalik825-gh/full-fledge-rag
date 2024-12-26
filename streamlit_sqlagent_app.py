from langchain.callbacks import StreamlitCallbackHandler
from langchain.sql_database import SQLDatabase
from langchain_groq import ChatGroq
from services.langchain_components.agents import create_sql_agent_executor
import streamlit as st
#import sqlite3

st.set_page_config(page_title="LangChain: Chat with SQL DB", page_icon="🦜")
st.title("🦜 LangChain: Chat with SQL DB")

groq_api_key = st.sidebar.text_input(label='Enter your groq api key',type='password')
llm=ChatGroq(model="mixtral-8x7b-32768",
                 temperature=0.5,
                 groq_api_key=groq_api_key,
                 streaming=True)

if not groq_api_key:
    st.info("Please add the groq api key")

@st.cache_resource(ttl="2h")
def configure_db(db_type,db_file=None,mysql_host=None,mysql_user=None,mysql_password=None,mysql_db=None):
    if db_type == 'upload sqlite db file':
        db = SQLDatabase.from_uri(f"sqlite:///{db_file.name}")
    else:
        if not (mysql_host and mysql_user and mysql_password and mysql_db):
            st.error("Please provide all MySQL connection details.")
            st.stop()
        db = SQLDatabase.from_uri(f"mysql+pymysql://{mysql_user}:{mysql_password}@{mysql_host}/{mysql_db}")
    return db

radio_options = ['upload sqlite db file','connect to localdb']
selected_option = st.sidebar.radio(label='Choose your database',options=radio_options)

if selected_option.index(selected_option)==0:
    db_file = st.sidebar.file_uploader('Upload sqlite db file',type=['db','sqlite','sqlite3'])
    if db_file:
        st.write(db_file)
        db = configure_db(
            db_type=selected_option,
            db_file=db_file)
else:
    mysql_host = st.sidebar.text_input('Enter your mysql host')
    mysql_user = st.sidebar.text_input('Enter your mysql user')
    mysql_password = st.sidebar.text_input('Enter your mysql password')
    mysql_db = st.sidebar.text_input('Enter your mysql db')
    db = configure_db(
            db_type=selected_option,
            mysql_host=mysql_host,
            mysql_user=mysql_user,
            mysql_password=mysql_password,
            mysql_db=mysql_db)
try:
    agent = create_sql_agent_executor(llm=llm,db=db)
except:
    pass

if "messages" not in st.session_state or st.sidebar.button('Clear chat history'):
    st.session_state['messages'] = [
        {"role":"assistant","content":"Hi,I'm a chatbot who can search the web. How can I help you?"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg['content'])

if query := st.chat_input(placeholder="Ask me anything about your db!"):
    st.session_state.messages.append({"role":"user","content":query})
    with st.chat_message("user"):
        st.write(query)
    with st.chat_message("assistant"):
        st_callback = StreamlitCallbackHandler(st.container())
        response = agent.run(query,callbacks=[st_callback])
        st.session_state.messages.append({"role":"assistant","content":response})
        st.write(response)



