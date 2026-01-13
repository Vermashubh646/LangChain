from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser 
from dotenv import load_dotenv

load_dotenv('../.env')

model=ChatGoogleGenerativeAI(model='gemini-2.5-flash-lite')

prompt=PromptTemplate(
    template='Write a Summary of the following text - \n{text}',
    input_variables=['text']
)

parser=StrOutputParser()

loader=PyPDFLoader(file_path='Comm_pdf.pdf')

docs=loader.load()

chain = prompt | model | parser

response=chain.invoke({'text':docs[1].page_content})

print(response)
print(docs[1].metadata)
print(docs[1].page_content)

# for more refer
# python.langchain.com/docs/integrations/document_loaders/#pdfs

# Simple, clean PDFs
# PyPDFLoader

# PDFs with tables/columns
# PDFPlumberLoader

# Scanned/image PDFs
# UnstructuredPDFLoader or AmazonTextractPDFLoader

# Need layout and image data
# PyMuPDFLoader

# Want best structure extraction
# UnstructuredPDFLoader









