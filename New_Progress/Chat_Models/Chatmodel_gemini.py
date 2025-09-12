from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv("../.env")

model = ChatGoogleGenerativeAI(model="gemini-2.5-pro")

result = model.invoke("What is the capital of India?")

print(result.content)