from llama_index.core import SimpleDirectoryReader
# Load VectorStoreIndex by selecting the splitter(chunk_size, chunk_overlap) and embedded model directly

from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import VectorStoreIndex

reader = SimpleDirectoryReader(input_dir=".")
docs = reader.load_data()
print(f"Count of Documents: {len(docs)}")
print(docs[0])


embed_model = OpenAIEmbedding()

node_parser = SentenceSplitter(chunk_size=256, chunk_overlap=20)
nodes = node_parser.get_nodes_from_documents(docs)
index = VectorStoreIndex(nodes, embed_model=embed_model, show_progress=True)

# 'similarity_top_k' refers to the number of top k chunks with the highest similarity.

base_retriever = index.as_retriever(similarity_top_k=5)

source_nodes = base_retriever.retrieve("Chapter")


# check source_nodes

for node in source_nodes:
    # print(node.metadata)
    print(f"---------------------------------------------")
    print(f"Score: {node.score:.3f}")
    print(node.get_content())
    print(f"---------------------------------------------\n\n")
