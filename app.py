from langchain_core.messages import HumanMessage, AIMessage
import os
from dotenv import dotenv_values
import streamlit as st
from agent import agent

# ENVs = dotenv_values()

try:
    ENVs = dotenv_values(".env")  # for dev env
    GROQ_API_KEY = ENVs["GROQ_API_KEY"]
except:
    ENVs = st.secrets  # for streamlit deployment
    GROQ_API_KEY = ENVs["GROQ_API_KEY"]


os.environ["GROQ_API_KEY"] = ENVs['GROQ_API_KEY']
os.environ["HUGGINGFACE_API_KEY"] = ENVs['HUGGINGFACE_API_KEY']

# create a Streamlit app

# configure the layout
st.set_page_config(
    page_title="Nyaya-GPT👩‍⚖️",
    page_icon="⚖️",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.title("Nyaya-GPT⚖️")
# st.header("I am your legal chatbot assistant")
initial_msg = """
    #### Welcome!!!  I am your legal assistant chatbot👩‍⚖️
    #### You can ask me any queries about the laws or constitution of India
    > NOTE: Currently I have only access to the Bharatiya Nyaya Sanhita (BNS) and the Indian Constitution. So, try to ask relevant queries only😇
    """
st.markdown(initial_msg)

# very important step, if we dont use st.session_state then it will not store the history in streamlits browser session
if "store" not in st.session_state:

    st.session_state.store = []

store = st.session_state.store

for message in store:
    if message.type == "ai":
        avatar = "👩‍⚖️"
    else:
        avatar = "🗨️"
    with st.chat_message(message.type, avatar=avatar):
        st.markdown(message.content)


# React to user input
if prompt := st.chat_input("What is your query?"):

    # Display user message in chat message container
    st.chat_message("user", avatar="🗨️").markdown(prompt)
    st.chat_message("⚖️").markdown("Thinking...")

    store.append(HumanMessage(content=prompt))
    response = AIMessage(content=agent(prompt))
    store.append(response)

    # Display assistant response in chat message container
    st.chat_message("assistant", avatar="👩‍⚖️").markdown(response.content)
