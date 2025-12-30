from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from dotenv import load_dotenv
import os

load_dotenv('../.env')

llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task='text-generation',
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
)

model = ChatHuggingFace(llm=llm)

parser= JsonOutputParser()

template= PromptTemplate(
    template="Give me the name, age, contribution and birth city of a random under rated figure related to physics \n {format_instruction}",
    input_variables=[''],
    partial_variables={'format_instruction': parser.get_format_instructions()}
)

# # as the filling is not given by user, so no runtime filling 
# prompt = template.format()

# response = model.invoke(prompt)
# print(response.content)
# final_response=parser.parse(response.content)
# print(final_response)


# another way
chain = template | model | parser

# you are always supposed to send a dictionary, if no i/p var then just empty dict
response=chain.invoke({})
print(response)