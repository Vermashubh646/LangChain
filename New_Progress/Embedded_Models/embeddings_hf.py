from langchain_huggingface import HuggingFaceEmbeddings


embedding = HuggingFaceEmbeddings(
    model_name='sentence-transformers/all-MiniLM-L6-v2',

    )

text = "Delhi is the capital of India"

vector = embedding.embed_query(text)
print(str(vector))

docs = [
    "The quick brown fox jumps over the lazy dog.",
    "Artificial intelligence is transforming the world.",
    "LangChain provides a framework for building LLM applications.",
    "Gemini is a family of generative AI models."
]

vector = embedding.embed_documents(docs)
print(str(vector))