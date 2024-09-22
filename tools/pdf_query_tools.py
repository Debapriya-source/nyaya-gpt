from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.agents import tool
from langchain_huggingface import HuggingFaceEmbeddings
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_groq import ChatGroq
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore


@tool
def indian_constitution_pdf_query(query: str) -> str:
    """Returns a related answer from the Indian Constitution PDF using semantic search from input query"""

    llm = ChatGroq(model="llama3-8b-8192")
    embeddings_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2")

    try:
        db = FAISS.load_local("db/faiss_index_constitution",
                              embeddings_model, allow_dangerous_deserialization=True)
    except:

        reader = PdfReader("tools/data/constitution.pdf")
        raw_text = ''
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                raw_text += text
        text_splitter = RecursiveCharacterTextSplitter(
            # separator="\n",
            chunk_size=800,
            chunk_overlap=400,
            # length_function=len,
        )
        texts = text_splitter.split_text(raw_text)
        # print(texts)

        db = FAISS.from_texts(texts, embeddings_model)  # cached_embedder)
        db.save_local("db/faiss_index_constitution")

    retriever = db.as_retriever(k=4)
    result = retriever.invoke(query)

    # print(result)
    return result
    # return docs


@tool
def indian_laws_pdf_query(query: str) -> str:
    """Returns a related answer from the "THE BHARATIYA NYAYA (SECOND) SANHITA, 2023" PDF which states all of the laws of India, using semantic search from input query"""

    llm = ChatGroq(model="llama3-8b-8192")
    embeddings_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2")

    try:
        db = FAISS.load_local("db/faiss_index_bns",
                              embeddings_model, allow_dangerous_deserialization=True)
    except:
        # if len(list(store.yield_keys())) == 0:
        # print("except")
        reader = PdfReader("tools/data/BNS.pdf")
        raw_text = ''
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                raw_text += text
        text_splitter = RecursiveCharacterTextSplitter(
            # separator="\n",
            chunk_size=800,
            chunk_overlap=400,
            # length_function=len,
        )
        texts = text_splitter.split_text(raw_text)
        # print(texts)

        db = FAISS.from_texts(texts, embeddings_model)
        db.save_local("db/faiss_index_bns")

    retriever = db.as_retriever(k=4)
    result = retriever.invoke(query)

    # print(result)
    return result
    # return docs
