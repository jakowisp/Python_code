import sys
import os
import time
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaLLM
from langchain_ollama import OllamaEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import UnstructuredRTFLoader

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = "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"
os.environ["LANGCHAIN_PROJECT"] = "My_project"
os.environ['USER_AGENT'] = 'myagent'



def process_input(question):
    model_local = OllamaLLM(model="mistral:v0.3")
    
    print("<-- Loading RAG files -->");
    starttime = time.perf_counter()
    loader = DirectoryLoader("./", glob="*.rtf", loader_cls=UnstructuredRTFLoader)
    docs = loader.load_and_split()
    docs_list = [item for sublist in docs for item in sublist]

    # This is where it "chunks" the data from the URLs
    #text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=7500, chunk_overlap=100)
    #text_splitter = CharacterTextSplitter(separator="\n\n", chunk_size=7500, chunk_overlap=100)
    #doc_splits = text_splitter.split_documents(docs_list)

    # Create a vector store using Chroma DB, our chunked data from the URLs, and the nomic-embed-text embedding model
    vectorstore = Chroma.from_documents(
        documents=docs,
        collection_name="rag-chroma",
        embedding=OllamaEmbeddings(model='nomic-embed-text'),
    )
    retriever = vectorstore.as_retriever()
    endtime = time.perf_counter()
    print("<-- Finished making index -->")
    print(endtime-starttime)
    print("\n")

    # Create a question / answer pipeline 
    rag_template = """Answer the question based only on the following context:
    {context}
    Question: {question}
    """
    rag_prompt = ChatPromptTemplate.from_template(rag_template)
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | rag_prompt
        | model_local
        | StrOutputParser()
    )
    # Invoke the pipeline
    return rag_chain.invoke(question)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 app.py '<question>'")
        sys.exit(1)

    question = sys.argv[1]
    answer = process_input(question)
    print(answer)
