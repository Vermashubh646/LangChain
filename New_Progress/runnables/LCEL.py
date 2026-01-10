from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
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

# LangChain Expression Language aka LCEL
# LCEL is all about composing chains using the pipe (|) operator, where the output of one component flows into the next.
# literally anywhere in place of runnable sequence you can use, even inside runnable parallel
chain = prompt1 | model | parser | prompt2 | model | parser

response=chain.invoke({'topic':'GenAI'})

print(response)