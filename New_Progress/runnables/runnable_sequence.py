from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv("../.env")

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

parser = StrOutputParser()

prompt1 = PromptTemplate(
    template="Tell a joke on {topic}",
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template="Explain the following Joke, the way is 'Joke is ...' Next Line 'explaination is ......' \n {text}",
    input_variables=['text']
)

chain = RunnableSequence(prompt1,model,parser,prompt2,model,parser)

response=chain.invoke({'topic':'GenAI'})

print(response)