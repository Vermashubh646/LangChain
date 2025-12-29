from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv('../.env')

model = ChatGoogleGenerativeAI(model='gemini-2.5-flash-lite')

messages=[
    SystemMessage(content='You are a professional and expert doctor.'),
    HumanMessage(content='Tell me about Tuberculosis.')
]

response=model.invoke(messages)

messages.append(response.content)

print(messages)