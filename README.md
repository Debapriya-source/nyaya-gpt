# Nyaya-GPT: Building Smarter Legal AI with ReAct + RAG
![thumbnail](https://github.com/user-attachments/assets/b10318a3-e886-49fb-96e7-9d0a5d1ac93d)
https://dev.to/debapriyadas/nyaya-gpt-building-smarter-legal-ai-with-react-rag-4l92

Let’s dive into some advanced AI concepts with a real-world problem like building a smart legal assistant, **Nyaya-GPT**. This project helps users to query legal documents like the **Indian Constitution** and **Bharatiya Nyaya Sanhita (BNS)** with precision. We'll explore concepts like **ReAct,** **RAG** (Retrieval-Augmented Generation) and **Vector Databases** and how they work together to push the boundaries of simple fact retrieval.

---

### **Problem Statement and Use Case**

Legal documents are vast, complex, and difficult to navigate without specialized knowledge. A traditional chatbot using basic retrieval systems can get overwhelmed with legal jargon or fail to pull the most relevant information from large documents. Nyaya-GPT solves this by combining **ReAct + RAG** to create a chatbot that not only retrieves legal facts but also reasons through them, offering more nuanced responses.

---

### **RAG (Retrieval-Augmented Generation)**: What Is It?

At its core, **RAG** combines **retrieval** with **generation**. In simpler terms, it looks for the most relevant information from a database or document (retrieval) and uses an LLM (Language Model) to formulate the response (generation).

However, basic RAG setups have limitations:

- **Only factual**: They retrieve facts using semantic search from a vector database without any kind of reasoning, which makes them okay for straight-up questions but lacking nuance.
- **No deeper understanding**: Naive RAG doesn’t really "understand" the query or refine the answers. It’s like asking a librarian for a book on a topic—they give you a book, but you still need to read and understand it yourself.
- **No memory**: These systems don’t remember previous queries or correct mistakes—they answer in a single shot, leaving no room for back-and-forth conversation.

---

### **Why ReAct + RAG?**

Now, this is where **ReAct (Reason + Act)** steps in to supercharge our naive RAG. Instead of just pulling up facts, **ReAct allows the model to reason** through a query and then **act** to retrieve relevant info in multiple steps. It uses a "think before you act" kind of approach, where the agent breaks down the query, performs actions (like retrieving data), and refines the answer before responding.

Here’s why **ReAct + RAG** is superior:

- **Query Understanding**: It doesn’t just do a blind search—it thinks about what you're asking. If the first attempt isn't great, it revises its actions.
- **Multi-step Reasoning**: Rather than fetching a single fact, it performs multiple steps to ensure the answer is accurate and contextually appropriate.
- **Error Handling and Memory**: This loop allows the system to handle mistakes and track the conversation, leading to better results over time. For example, if the prompt contains any kind of typo or it is vague or incomplete in some manner, the reasoning loop will try to handle that as per its capability.

---

### **The Role of a Vector Database**

To make retrieval smarter, Nyaya-GPT uses a **vector database**. But what exactly is a vector database?

Instead of storing data as simple text, vector databases store it as **embeddings**—numerical representations of meaning in the text. For Nyaya-GPT, this means breaking down the legal documents into chunks and storing them as vectors. When you ask a question, the system converts your query into a vector and searches for semantically similar vectors from the stored chunks.

**Why is this important?**

- **Efficient Semantic Search**: A vector database helps the system understand the meaning behind words, not just match keywords.
- **Scalability**: As new legal documents are added, the system can handle larger datasets efficiently.
- **Relevance**: It retrieves the most relevant chunks of information, which are then used by the LLM to craft a detailed response.

For instance, if you ask Nyaya-GPT about “fundamental rights,” it doesn’t just look for exact keywords—it searches for related legal concepts and sections, thanks to the vector database.

---

### **Workflow of Nyaya-GPT**


![Nyaya-GPT workflow of ReAct+RAG loop](https://dev-to-uploads.s3.amazonaws.com/uploads/articles/gif2o9wfgm49cakya4br.png)

Here’s how everything comes together in Nyaya-GPT’s workflow:

1. **User Query**: An user submits a legal question like, *“What are the key provisions of the Indian Constitution?”*
2. **ReAct Loop**: The system analyzes the question and determines whether it requires retrieval or reasoning.
3. **RAG & Vector Database**: It fetches relevant legal text from the FAISS(Facebook AI Similarity Search) vector database, using semantic search based on embeddings.
4. **Thought-Action Cycle**: The agent reasons through the query and refines the result using the information retrieved.
5. **Final Answer**: The system synthesizes the retrieved information into a detailed, accurate response.

---

### **Code Walkthrough: How Nyaya-GPT Implements ReAct + RAG**

Let’s dive into some code snippets to see how Nyaya-GPT brings these concepts to life.

**Step 1: Agent Creation Using ReAct and Tools**

The heart of Nyaya-GPT is the **ReAct agent**, which handles both reasoning and tool invocation. Below is the key function from `agent.py`:

```python
from langchain_groq import ChatGroq
from langchain.agents import create_react_agent, AgentExecutor
from tools.pdf_query_tools import indian_constitution_pdf_query, indian_laws_pdf_query

def agent(query: str):
    LLM = ChatGroq(model="llama3-8b-8192")
    tools = [indian_constitution_pdf_query, indian_laws_pdf_query]
    prompt_template = get_prompt_template()

    agent = create_react_agent(LLM, tools, prompt_template)

    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False, handle_parsing_errors=True)
    result = agent_executor.invoke({"input": query})

    return result["output"]

```

This function sets up the **LLM**, along with the necessary tools for interacting with the vector database. When a query is received, the agent invokes the ReAct loop, reasoning through the query and determining if it needs to retrieve any documents using the tools.

---

**Step 2: PDF Query Tools and Vector Search**

Nyaya-GPT uses **FAISS** to store document embeddings and perform **semantic search**. In `pdf_query_tools.py`, this function loads the Indian Constitution as a vector database and retrieves relevant sections based on the query:

```python
from langchain_community.vectorstores import FAISS
from PyPDF2 import PdfReader

def indian_constitution_pdf_query(query: str) -> str:
    embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

    try:
        db = FAISS.load_local("db/faiss_index_constitution", embeddings_model)
    except:
        reader = PdfReader("tools/data/constitution.pdf")
        raw_text = ''.join(page.extract_text() for page in reader.pages if page.extract_text())

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=400)
        texts = text_splitter.split_text(raw_text)
        db = FAISS.from_texts(texts, embeddings_model)
        db.save_local("db/faiss_index_constitution")

    retriever = db.as_retriever(k=4)
    return retriever.invoke(query)

```

This snippet demonstrates how the Indian Constitution PDF is loaded, processed into text chunks, and embedded into the FAISS database. When a user queries Nyaya-GPT, the system searches through these embeddings to find the most relevant text.

---

**Step 3: Streamlit Interface**

The **Streamlit app** serves as the front end for interacting with Nyaya-GPT. Users can input their queries directly into the interface, which then calls the agent function to retrieve and display answers.

```python
import streamlit as st
from agent import agent

st.title("Nyaya-GPT⚖️")

if "store" not in st.session_state:
    st.session_state.store = []

store = st.session_state.store

if prompt := st.chat_input("What is your query?"):
    st.chat_message("user").markdown(prompt)
    response = agent(prompt)
    store.append(response)
    st.chat_message("assistant").markdown(response.content)

```
This interface provides a simple yet effective way to interact with the chatbot, allowing users to query legal documents and receive answers.

---
> Checkout this article for a detailed walkthrough to on building chatbot using LLMs powered by Groq and Streamlit: [https://dev.to/debapriyadas/create-an-end-to-end-personalised-ai-chatbot-using-llama-31-and-streamlitpowered-by-groq-api-3i32](https://dev.to/debapriyadas/create-an-end-to-end-personalised-ai-chatbot-using-llama-31-and-streamlitpowered-by-groq-api-3i32)

---

### **Conclusion**

Nyaya-GPT demonstrates the power of combining **RAG** and **ReAct** to build a sophisticated legal assistant capable of answering complex legal queries. By leveraging **FAISS vector databases** for efficient retrieval and ensuring that the model reasons through its responses, the system offers a more reliable and scalable solution than traditional approaches.

For more information and to access the code, check out the repository:

[Nyaya-GPT GitHub Repository](https://github.com/Debapriya-source/nyaya-gpt)

**Additional Resources:**

- [LangChain Documentation](https://docs.langchain.com/)
- [FAISS Vector Store](https://faiss.ai/)
- [Groq API](https://groq.com/)
- [Streamlit](https://streamlit.io/)

This combination of structured reasoning, powerful retrieval, and an intuitive user interface creates an efficient legal research tool that can assist users in navigating complex laws with ease.
Edit the documents and the codebase to create your own personalized assistant.

---
