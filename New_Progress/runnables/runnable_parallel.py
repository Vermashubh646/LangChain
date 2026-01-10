from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel,RunnableSequence
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv("../.env")

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

parser = StrOutputParser()

prompt1 = PromptTemplate(
    template="Write 5 postive of {topic}",
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template="Write 5 negatives of {topic}",
    input_variables=['topic']
)

chain = RunnableParallel({
    'pros':RunnableSequence(prompt1,model,parser),
    'cons':RunnableSequence(prompt2,model,parser)
})

response=chain.invoke({'topic':'GenAI'})

print(response['pros'],'\n\n')
print(response['cons'])