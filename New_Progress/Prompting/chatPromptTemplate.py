from langchain_core.prompts import ChatPromptTemplate

chatTemplate = ChatPromptTemplate([
    'system','You are a Helpful {domain} Expert.',
    'human','Explain in Simple terms what is {topic}'
])

prompt = chatTemplate.invoke({'domain':'cricket','topic':'LBW'})

print(prompt)
