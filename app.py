from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import (
    BaseChatMessageHistory,
    InMemoryChatMessageHistory,
)
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
import os
from dotenv import dotenv_values
import streamlit as st


GROQ_API_KEY = dotenv_values()

os.environ["GROQ_API_KEY"] = GROQ_API_KEY['GROQ_API_KEY']


model = ChatGroq(model="llama3-8b-8192")

# create a Streamlit app

# configure the layout
st.set_page_config(
    page_title="LangChain Groq with Llama3",
    page_icon=":robot:",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.title("LangChain-Groq-Llama-3")


# using normal list to implement chat history
# # Initialize chat history
# if "messages" not in st.session_state:
#     st.session_state.messages = []

# # Display chat messages from history on app rerun
# for message in st.session_state.messages:
#     with st.chat_message(message["role"]):
#         st.markdown(message["content"])

# # React to user input
# if prompt := st.chat_input("What is your message?"):
#     # Display user message in chat message container
#     st.chat_message("user").markdown(prompt)
#     # Add user message to chat history
#     st.session_state.messages.append({"role": "user", "content": prompt})

#     print(st.session_state.messages)
#     messages = [
#         {"role": "system", "content": "you are a footballer"
#          },
#         {"role": "assistant", "content": "Hi whatsup!"},
#         *st.session_state.messages
#     ]
#
#     response = model.invoke(messages)

#     # Display assistant response in chat message container
#     with st.chat_message("assistant"):
#         st.markdown(response.content)
#     # Add assistant response to chat history
#     st.session_state.messages.append(
#         {"role": "assistant", "content": response.content})


# using InMemoryChatMessageHistory to implement chat history
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        print("creating new session", session_id, st.session_state.store)
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]
# # Initialize chat history


# very important step, if we dont use st.session_state then it will not store the history in streamlits browser session
if "store" not in st.session_state:
    st.session_state.store = {}

store = st.session_state.store

config = {"configurable": {"session_id": "abc2"}}
with_message_history = RunnableWithMessageHistory(
    model, get_session_history)

# display chat history
session_history = get_session_history(config['configurable']["session_id"])
# write the chat history in the beginning
# to be writen
# print(session_history)
for message in session_history.messages:
    with st.chat_message(message.type):
        st.markdown(message.content)


# React to user input
if prompt := st.chat_input("What is your message?"):
    # print(store)
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)

    # with message history send the prompt
    response = with_message_history.invoke(
        [HumanMessage(content=prompt)],
        config=config,
    )
    # print(store)
    # Display assistant response in chat message container
    st.chat_message("assistant").markdown(response.content)
