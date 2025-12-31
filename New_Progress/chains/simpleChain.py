from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
import os
from dotenv import load_dotenv

load_dotenv('../.env')


llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
)

model = ChatHuggingFace(llm=llm)

template1 = PromptTemplate(
    template="Write 5 lines about {topic}",
    input_variables=['topic']
)

parser=StrOutputParser()

chain = template1 | model | parser 

response=chain.invoke({'topic':'Engineering'})

print(response)

chain.get_graph().print_ascii()



