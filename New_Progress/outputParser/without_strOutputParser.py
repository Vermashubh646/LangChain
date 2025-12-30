from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
import os

load_dotenv('../.env')

llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
)

model = ChatHuggingFace(llm=llm)

template1 = PromptTemplate(
    template="Write down an detailed report on {topic}",
    input_variables=['topic']
)

template2 = PromptTemplate(
    template="write a  5 lines summary on the following text.\n {text} ",
    input_variables=['text']
)

prompt1=template1.invoke({'topic':'Kantara Chapter 1 movie'})
response1=model.invoke(prompt1)
prompt2= template2.invoke({'text':response1.content})
response2 = model.invoke(prompt2)

print(response2.content)