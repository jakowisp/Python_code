from llama_index.core import SimpleDirectoryReader
from llama_index.core import VectorStoreIndex

reader = SimpleDirectoryReader(input_dir=".", required_exts=[".txt",".rtf",".docx"])
docs = reader.load_data()
print(f"Count of Documents: {len(docs)}")
print(docs[0])

# 1. Load VectorStoreIndex directly from Documents
index = VectorStoreIndex.from_documents(docs, show_progress=True)
