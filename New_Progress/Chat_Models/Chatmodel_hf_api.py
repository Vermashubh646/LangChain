from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os

load_dotenv("../.env")

llm= HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V3.1",
    task="text-generation",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
)

model = ChatHuggingFace(llm=llm)

result = model.invoke(
    '''
        What is Captial of India?
    '''
)

print(result.content)

