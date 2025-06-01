from fastapi import FastAPI, Form, UploadFile, File
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA #This line help in searching from document and gives answer
from langchain_ollama import OllamaLLM
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain_core.prompts import PromptTemplate
from langchain.agents.agent_types import AgentType
# Removed import of BaseConversationalRetrievalChain due to ImportError
# from langchain_community.chains import BaseConversationalRetrievalChain
# Removed import of Neo4jGraph as it was only used in the removed /kg-rag endpoint
# from langchain_community.graphs import Neo4jGraph
import sqlite3
import os


app = FastAPI()

llm = OllamaLLM(model="llama3.2:latest")
embedding_model = OllamaEmbeddings(model="mxbai-embed-large:latest")

# @app.on_event("startup")
# async def startup_event():
#     app.state.temp_file_path = None  # Initialize temp_file_path during startup

@app.post("/ask")
async def ask_question(file:UploadFile=File(...), query:str=Form(...)):
    contents= await file.read()
    temp_path = f"temp_{file.filename}"
    # if os.path.exists(temp_path):
    #     with open(temp_path, "rb") as f:
    #        existing_contents=f.read()
    #        loader = PyPDFLoader(existing_contents)
    # else:
    with open(temp_path, "wb") as f:
            f.write(contents)
    loader = PyPDFLoader(temp_path)
    #noting the path of the file that is uploaded
    app.state.temp_file_path = temp_path
    print("Temp file path set to**********************", app.state.temp_file_path)
    docs = loader.load()
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=50) #small chunks as my system storage is not sufficient for llama3.2
    print("Documents loaded successfully**********************", docs)
    chunks = text_splitter.split_documents(docs)
    print("Chunks created successfully**********************", chunks)

    vectorstore= FAISS.from_documents(chunks, embedding_model)
    print("Vectorstore created successfully**********************", vectorstore)
    retriver = vectorstore.as_retriever()
    print("Retriever created successfully**********************", retriver)

    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriver)
    print("QA chain created successfully**********************", qa_chain)
    result = qa_chain.run(query)

    # os.remove(temp_path)

    return {"query":query,
            "file":file.filename,
        "response": result}


@app.post("/ask-sql")
async def ask_sql(file:UploadFile=File(...) ,query:str=Form(...)):
    print("Received file **********************", f"temp_{file.filename}")
    temp_file_path = f"temp_{file.filename}"
    print("Temp file path**********************************", temp_file_path)
    # if not temp_file_path or not os.path.exists(temp_file_path):
    #     return {"error": "No file uploaded or file not found."}
    file_name = temp_file_path.split('.')[0]  # Extract the filename without extension
    print("File name without extension **********************", file_name)
    conn = sqlite3.connect(f"{file_name}.db")
    try:
        cursor = conn.cursor()
        cursor.execute(f'''CREATE TABLE IF NOT EXISTS {file_name}(
                    id Integer PRIMARY KEY AUTOINCREMENT,
                    questions TEXT,
                    answers TEXT,
                    filename TEXT)''')
        resume_db=SQLDatabase.from_uri(f"sqlite:///{file_name}.db")
        agent = create_sql_agent(
            llm=llm,
            db = resume_db,
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            prompt=PromptTemplate.from_template(
                 """You are a helpful assistant that answers questions from the file. 
            Use the tools provided to answer the question based on the information in the database.

            Question: {{query}}
            Tools: {tools}
            Tool Names: {tool_names}
            Agent Scratchpad: {agent_scratchpad}
            When answering, follow this format:
            Thought: [Your thought process]
            Action: [The action you will take, e.g., tool name]
            Action Input: [Input for the action]
            Observation: [Result of the action]
            Final Answer: [Your final answer based on observations]"""
            ),
            verbose=True

        )
        try:
            result = agent.run(input = query, handle_parsing_errors=True)
        except Exception as e:
             print(f"Error running agent: {e}")



        cursor.execute(f"INSERT INTO {file_name} (questions, answers, filename) VALUES(?,?,?)",(query, result, file_name))
        conn.commit()
        
        return {
            "query": query,
            "response": result
        }
    finally:
        conn.close()

# Removed /kg-rag endpoint due to missing BaseConversationalRetrievalChain class
# @app.get("/kg-rag")
# async def kgrag(question:str):
#      graph = Neo4jGraph(
#           url="bolt://localhost:7687",
#           username="neo4j",
#           password="test",
#         )
#      chain = BaseConversationalRetrievalChain.from_llm(llm=llm, graph = graph, verbose=True)
#      result = chain.run(question)
#      return {
#           "query": question,
#           "response": result
#      }
