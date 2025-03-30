from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

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
        context += "\nUser: {user_input}\nAssitant: {result}"
        

model = OllamaLLM(model="sonja")
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model
handle_conversation()
