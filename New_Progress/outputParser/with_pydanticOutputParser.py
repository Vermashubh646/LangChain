from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from typing import Annotated
from langchain_core.output_parsers import PydanticOutputParser
from dotenv import load_dotenv
import os

load_dotenv('../.env')

llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task='text-generation',
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
)

model = ChatHuggingFace(llm=llm)

class Person(BaseModel):

    name:Annotated[str, Field(...,description="Name of the Person")]
    age: Annotated[int, Field(...,description="Age of the Person. Upto Date of Death if the person has died.")]
    birth_city:Annotated[str, Field(...,description="City of Birth of the Person")]
    contribution:Annotated[str, Field(...,description="Contribution of the Person to the topic in one line")]

parser= PydanticOutputParser(pydantic_object=Person)

template= PromptTemplate(
    template="Give me the name, age, contribution and birth city of a random under rated figure related to {topic} \n {format_instruction}",
    input_variables=['topic'],
    partial_variables={'format_instruction': parser.get_format_instructions()}
)

# print(template.invoke({'physics'}))

chain = template | model | parser

response=chain.invoke({'topic':'history of india'})
print(response)