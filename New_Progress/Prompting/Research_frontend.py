import streamlit as st
from langchain_core.prompts import load_prompt
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import time

load_dotenv("../.env")

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

st.title("Research Paper Summarizer AI")

paper_input = st.selectbox(
    "Choose the Paper",
    [
    "Attention Is All You Need",
    "ImageNet Classification with Deep Convolutional Neural Networks",
    "Deep Residual Learning for Image Recognition",
    "Generative Adversarial Nets",
    "A Few Useful Things to Know About Machine Learning",
    "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"
    ]
    )

style_input = st.selectbox(
    "Choose Explaination Type",
    [
    "Code-Oriented",
    "Algorithm-Centric",
    "Math-Driven",
    "Architecture-Level",
    "Experiment & Results-Focused"
    ]

)

length_input = st.selectbox(
    "Choose Length",
    [
    "Very Short Answer",
    "Short Answer",
    "Paragraph Answer",
    "Long Answer",
    "Essay-Type Answer"
    ]
)

template = load_prompt('prompt_summarize.json')
# another way
# prompt = template.invoke({
#     'paper_input':paper_input,
#     'style_input':style_input,
#     'length_input':length_input
# })

@st.cache_data(show_spinner=False)
def summarize_paper(paper, style, length):
    chain = template | model
    response = chain.invoke({
        'paper_input': paper,
        'style_input': style,
        'length_input': length
    })
    return response.content

if st.button('Summarize'):
    result = summarize_paper(paper_input, style_input, length_input)
    st.write(result)
 

