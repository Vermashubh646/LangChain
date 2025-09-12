from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv("../.env")

# Reduce to 10 dimensions
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004"
    # the below part don't works
    # output_dimensionality=10,
)

# Embed a query
vector = embeddings.embed_query("hello, world!", output_dimensionality=10)
print(len(vector), vector)

docs = [
    "The quick brown fox jumps over the lazy dog.",
    "Artificial intelligence is transforming the world.",
    "LangChain provides a framework for building LLM applications.",
    "Gemini is a family of generative AI models."
]

vectors = embeddings.embed_documents(docs, output_dimensionality=10)

# Check the output
print(f"Successfully embedded {len(vectors)} documents.")
print(f"Dimension of the first vector: {len(vectors[0])}")
print(f"First 5 elements of the first vector: {vectors}")