from langchain_community.document_loaders import WebBaseLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser 
from dotenv import load_dotenv

load_dotenv('../.env')

model=ChatGoogleGenerativeAI(model='gemini-2.5-flash-lite')

prompt=PromptTemplate(
    template='Answer the following question -\n{quest} from the following text - \n{text}',
    input_variables=['quest','text']
)

parser=StrOutputParser()

url='https://www.flipkart.com/apple-macbook-air-m2-16-gb-256-gb-ssd-macos-sequoia-mc7x4hn-a/p/itmdc5308fa78421?pid=COMH64PY76CJKBYU&lid=LSTCOMH64PY76CJKBYUOL7TOK&marketplace=FLIPKART&store=6bo%2Fb5g&spotlightTagId=default_BestsellerId_6bo%2Fb5g&srno=b_1_4&otracker=browse&fm=organic&iid=3d705fdf-8e24-4dfb-864c-234e977f7de1.COMH64PY76CJKBYU.SEARCH&ppt=None&ppn=None&ssid=zmq9fibaqo0000001768227976039'

loader=WebBaseLoader(url)

docs=loader.load()

chain = prompt | model | parser

response=chain.invoke({'quest':'What is the dimension, product type and name of product?','text':docs[0].page_content})
print(response)