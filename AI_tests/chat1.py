from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

template = """
Answer the question below.

Here is the conversation history: {context}

Question: {question}

Answer:
    """

model = OllamaLLM(model="mistral:v0.3")
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model
result = chain.invoke({"context":"","question":"what is artificial intelegence?"})
print(result)
