from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnablePassthrough
from langchain_classic.schema.runnable import RunnableLambda
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv("../.env")

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

parser = StrOutputParser()

prompt1 = PromptTemplate(
    template="Write a 50 word paragraph on {topic}, don't give any sort of title or starting line like here is your para, directly give para",
    input_variables=['topic']
)

def word_count(text):
    return len(text.split())

para_chain=RunnableSequence(prompt1,model,parser)

parallel_chain = RunnableParallel({
    'para':RunnablePassthrough(),
    'count':RunnableLambda(word_count)
})
# below also works 
# parallel_chain = RunnableParallel({
#     'para':RunnablePassthrough(),
#     'count':RunnableLambda(lambda x: len(x.split()))
# })

fullchain= RunnableSequence(para_chain,parallel_chain)

response = fullchain.invoke({'topic':'Prime Minister'})

print(response['para'],'\n\n')
print(response['count'],'\n\n')