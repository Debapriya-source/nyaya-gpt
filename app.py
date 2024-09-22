from langchain_core.messages import HumanMessage, AIMessage
import os
from dotenv import dotenv_values
import streamlit as st
from agent import agent

ENVs = dotenv_values()

os.environ["GROQ_API_KEY"] = ENVs['GROQ_API_KEY']
os.environ["HUGGINGFACE_API_KEY"] = ENVs['HUGGINGFACE_API_KEY']

# create a Streamlit app

# configure the layout
st.set_page_config(
    page_title="Chat with Indian Constitution",
    page_icon=":robot:",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.title("Chat with Indian Constitution")


# very important step, if we dont use st.session_state then it will not store the history in streamlits browser session
if "store" not in st.session_state:
    st.session_state.store = []

store = st.session_state.store

for message in store:
    with st.chat_message(message.type):
        st.markdown(message.content)


# React to user input
if prompt := st.chat_input("What is your message?"):

    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    st.chat_message(":robot:").markdown("Thinking...")

    store.append(HumanMessage(content=prompt))
    response = AIMessage(content=agent(prompt))
    store.append(response)

    # Display assistant response in chat message container
    st.chat_message("assistant").markdown(response.content)
