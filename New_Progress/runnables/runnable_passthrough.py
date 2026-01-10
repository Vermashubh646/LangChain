from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnablePassthrough
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

joke_chain=RunnableSequence(prompt1,model,parser)

parallel_chain = RunnableParallel({
    'joke':RunnablePassthrough(),
    'explain':RunnableSequence(prompt2,model,parser)
})

fullchain= RunnableSequence(joke_chain,parallel_chain)

response = fullchain.invoke({'topic':'Prime Minister'})

print(response['joke'],'\n\n')
print(response['explain'],'\n\n')