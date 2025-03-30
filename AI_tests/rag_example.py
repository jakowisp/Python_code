import os
from langchain_ollama import OllamaLLM
from dotenv import load_dotenv
from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import UnstructuredRTFLoader
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import create_retrieval_chain
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain


os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = "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"
os.environ["LANGCHAIN_PROJECT"] = "My_project"

#Loading The LLM (Language Model)
llm = OllamaLLM(model="mistral:v0.3", base_url="http://127.0.0.1:11434")
#Setting Ollama Embeddings
embed_model = OllamaEmbeddings(
        model="mistral:v0.3",
    base_url='http://127.0.0.1:11434'
)
#Loading Text
text = """
    In the lush canopy of a tropical rainforest, two mischievous monkeys, Coco and Mango, swung from branch to branch, their playful antics echoing through the trees. They were inseparable companions, sharing everything from juicy fruits to secret hideouts high above the forest floor. One day, while exploring a new part of the forest, Coco stumbled upon a beautiful orchid hidden among the foliage. Entranced by its delicate petals, Coco plucked it and presented it to Mango with a wide grin. Overwhelmed by Coco's gesture of friendship, Mango hugged Coco tightly, cherishing the bond they shared. From that day on, Coco and Mango ventured through the forest together, their friendship growing stronger with each passing adventure. As they watched the sun dip below the horizon, casting a golden glow over the treetops, they knew that no matter what challenges lay ahead, they would always have each other, and their hearts brimmed with joy.
    """


loader = UnstructuredRTFLoader( "BoS_dd.rtf", mode="elements", strategy="fast",)
docs = loader.load()

#Splitting Text into Chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=128)
chunks = text_splitter.split_text(text)
#Creating a Vector Store (Chroma) from Text
vector_store = Chroma.from_texts(chunks, embed_model)
#Creating a Retriever
retriever = vector_store.as_retriever()
#Creating a Retrieval Chain
chain = create_retrieval_chain(combine_docs_chain=llm,retriever=retriever)
#Retrieval-QA Chat Prompt
retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
#Combining Documents
combine_docs_chain = create_stuff_documents_chain(
    llm, retrieval_qa_chat_prompt
)
#Final Retrieval Chain
retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)    
#Invoking the Retrieval Chain
response = retrieval_chain.invoke({"input": "Tell me the monkey names"})
print(response['answer'])
