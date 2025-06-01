Traditional RAG:-
            LangChain + Ollama LLM
            FAISS for vector storage     
            FastAPI to expose an endpoint
            PDF ingestion + real-time querying
            Upload a PDF + query via '/ask'
            You need to have ollama and your desired model to be installed in your system to use it
            Splits and embeds the PDF using ollamaembedding
            FAISS for fast similarity search over embedded chunks based on our query asked.
            Runs RAG LangChain QA pipeline
            RetrivalQA takes the vector embedded data and the chain inputs the query to get the relavant answer using local Ollama LLM


            
Text2SQL:-
          Upload a PDF + query via '/ask-sql'
          This actually communicates with the sql database.
          I am using ReAct(reason+act) where the agent:-
                                                      where the agent:
                                                                    Reads my question
                                                                    Figures out what tool to use (in this case, SQL)
                                                                    Generates a query (like SELECT * FROM filename)
                                                                    Executes the query
                                                                    Returns the answer — all in one shot, without seeing any examples before (zero-shot)
                                                                    Now the sqlite stores the question and llm generated answer dynamically



Knowledge-Graph:-
                A Knowledge Graph (KG) is a structured way to represent entities (people, places, topics) and the relationships between them.
                Each piece of data becomes a node (entity) or an edge (relationship), and we can query these relationships.
                In KG-RAG, you let the LLM query a structured semantic network (your KG)
                Get the result from '/kg-rag'
                Connects to a Neo4j knowledge graph (bolt://localhost:7687)
                Uses LangChain's KGQAChain to query the graph using your LLM





To run this full project:-
                        Clone it, 
                        pip install -r requirements.txt
                        uvicorn main:app --reload 
