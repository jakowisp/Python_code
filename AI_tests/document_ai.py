from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import TextLoader

loader = DirectoryLoader("../", glob="**/*.rtf", loader_cls=TextLoader)
docs = loader.load()

template = """
Answer the question below.

Here is the conversation history: {context}

Question: {question}

Answer:
    """

def handle_conversation():
    context= ""
    print("Welcome to the explicit AI chatbot")
    while True:
        user_input=input("You: ")
        if user_input.lower() == "exit":
            break
        result = chain.invoke({"context":context,"question":user_input})
        print(result)
        context += "\nUser: {user_input}\nAI: {result}"
        

model = OllamaLLM(model="mistral:v0.3")
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model
handle_conversation()
