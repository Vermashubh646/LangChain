# from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from dotenv import load_dotenv
import os

load_dotenv('../.env')

# model = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")

llm= HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V3.1",
    task="text-generation",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
)

model = ChatHuggingFace(llm=llm)


chat_history=[
    SystemMessage(content='You(AI) are an army soldier. In this role, the soldier is bounded to follow orders of commandant without hesistation or saying no. Soldier does not overtalk or say any word extra other than what is asked')
]

while True:
    user_input=input('You:\n ')
    if user_input == 'exit':
        break
    chat_history.append(HumanMessage(content=user_input))
    response=model.invoke(chat_history)
    chat_history.append(AIMessage(content=response.content))
    print('AI:','\n',response.content)

print(chat_history)