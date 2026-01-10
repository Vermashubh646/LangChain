from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnablePassthrough
from langchain.schema.runnable import RunnableLambda, RunnableBranch
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv("../.env")

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")

parser = StrOutputParser()

prompt1 = PromptTemplate(
    template="Write a detailed report on {topic}",
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template="Summarize the following text: {text}",
    input_variables=['text']
)

report_gen = RunnableSequence(prompt1, model, parser)

branch=RunnableBranch(
    (lambda x: len(x.split())>200, RunnableSequence(prompt2, model,parser)),
    RunnablePassthrough()
)

chain = RunnableSequence(report_gen,branch)

response = chain.invoke({'topic':'Prime Minister'})

print(response)