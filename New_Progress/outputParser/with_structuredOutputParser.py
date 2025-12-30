from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from dotenv import load_dotenv
import os

load_dotenv('../.env')

llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task='text-generation',
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
)

model = ChatHuggingFace(llm=llm)

schema=[
    ResponseSchema(name='Fact_1',description='1st fact about the topic'),
    ResponseSchema(name='Fact_2',description='2nd fact about the topic'),
    ResponseSchema(name='Fact_3',description='3rd fact about the topic')
]

parser= StructuredOutputParser.from_response_schemas(schema)

template= PromptTemplate(
    template="Give me 3 facts of a random under rated figure related to field of {topic} \n {format_instruction}",
    input_variables=['topic'],
    partial_variables={'format_instruction': parser.get_format_instructions()}
)

chain = template | model | parser

response=chain.invoke({'topic':'physics'})
print(response)