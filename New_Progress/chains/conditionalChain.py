from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Annotated, Literal
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema.runnable import RunnableBranch, RunnableLambda
from dotenv import load_dotenv

load_dotenv('../.env')


model = ChatGoogleGenerativeAI(model='gemini-2.5-flash')


class Feedback(BaseModel):
    sentiment:Annotated[Literal['positive','negative'], Field(description='Semtiment of the text given') ]

parser=StrOutputParser()

parser2=PydanticOutputParser(pydantic_object=Feedback)

prompt1= PromptTemplate(
    template='Give the sentiment of the following feedback \n {feedback} \n {format_instruction}',
    input_variables=['feedback'],
    partial_variables={'format_instruction':parser2.get_format_instructions()}
)

classifier_chain = prompt1 | model | parser2

prompt2 = PromptTemplate(
template='Write an appropriate response to this positive feedback \n {feedback}',
input_variables=['feedback']
)

prompt3 = PromptTemplate(
template='Write an appropriate response to this negative feedback \n {feedback}',
input_variables=['feedback']
)

branch_chain = RunnableBranch(
(lambda x: x.sentiment == 'positive', prompt2 | model | parser),
(lambda x: x.sentiment == 'negative', prompt3 | model | parser),
RunnableLambda(lambda x: "could not find sentiment")
)

chain = classifier_chain | branch_chain

print(chain.invoke({'feedback': 'This is a beautiful phone'}))

chain.get_graph().print_ascii()