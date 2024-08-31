from langchain.memory import ConversationBufferMemory
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain.tools import DuckDuckGoSearchRun
from langchain.agents import AgentType, initialize_agent
from langchain_core.tools import Tool
from langchain_groq import ChatGroq 
from dotenv import load_dotenv
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import Docx2txtLoader
import os
import streamlit as st

load_dotenv()

def create_document_tool(docs):
    # Create embeddings
    model_name = "all-MiniLM-L6-v2"  
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    llm = ChatGroq(model="llama-3.1-8b-instant")

    
    # Create vector store
    vector_store = FAISS.from_documents(docs, embeddings)
    
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
    
    return Tool(
        name="Document QA",
        func=qa_chain.run,
        description="Useful for when you need to answer questions about the uploaded documents"
    )

def create_research_assistant(uploaded_files):
    
    llm = ChatGroq(model="llama-3.1-8b-instant")

    search = GoogleSerperAPIWrapper()
    ddg_search = DuckDuckGoSearchRun()
    
    tools = [
        Tool(
            name="Search",
            func=search.run,
            description="Useful for when you need to answer questions about current events or the current state of the world"
        ),
        Tool(
            name="DuckDuckGo Search",
            func=ddg_search.run,
            description="Useful for when you need to find information from the web"
        )
    ]

    
    docs = []
    for file in uploaded_files:
        # Save the uploaded file temporarily
        with open(os.path.join("/tmp", file.name), "wb") as f:
            f.write(file.getbuffer())

        if file.name.endswith('.pdf'):
            loader = PyPDFLoader(os.path.join("/tmp", file.name))
        elif file.name.endswith('.docx'):
            loader = Docx2txtLoader(os.path.join("/tmp", file.name))
        elif file.name.endswith('.txt'):
            loader = TextLoader(os.path.join("/tmp", file.name))
        else:
            st.warning(f"Unsupported file type: {file.name}")
            continue
        docs.extend(loader.load())

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = text_splitter.split_documents(docs)

    if split_docs:
        doc_tool = create_document_tool(split_docs)
        tools.append(doc_tool)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    agent = initialize_agent(
        tools, 
        llm, 
        agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION, 
        verbose=True, 
        memory=memory
    )

    return agent


