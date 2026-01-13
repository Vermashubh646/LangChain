from langchain_community.document_loaders import TextLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser 
from dotenv import load_dotenv

load_dotenv('../.env')

model=ChatGoogleGenerativeAI(model='gemini-2.5-flash-lite')

prompt=PromptTemplate(
    template='Write a Sumary of the following Poem - \n{text}',
    input_variables=['text']
)

parser=StrOutputParser()

loader = TextLoader('krishna.txt')

docs= loader.load()

chain = prompt | model | parser

response=chain.invoke({'text':docs[0].page_content})

print(response)