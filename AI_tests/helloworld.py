from langchain_ollama import OllamaLLM

model = OllamaLLM(model="mistral:v0.3")
result = model.invoke(input="hello world")
print(result)
