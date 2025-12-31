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
    template="Write down an detailed report on {topic}",
    input_variables=['topic']
)

template2 = PromptTemplate(
    template="write a  5 lines summary on the following text.\n {text} ",
    input_variables=['text']
)

parser=StrOutputParser()

chain = template1 | model | parser | template2 | model | parser

response=chain.invoke({'topic':'unemployment in india'})

print(response)

chain.get_graph().print_ascii()



