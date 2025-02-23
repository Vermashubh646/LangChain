from dotenv import load_dotenv
import os

# Load environment variables from config.env
load_dotenv("environment_tut1.env")


langsmith_api_key = os.getenv("LANGSMITH_API_KEY")
gemini_api_key = os.getenv("gemini_API_KEY")

from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=gemini_api_key)




import streamlit as st 

from langchain.memory import ConversationBufferMemory
# Initialize memory to store the last 15 messages
memory = ConversationBufferMemory(k=15)

def health_care_chatbot(query):
    system_prompt = (
        "You are an expert AI assistant specialized in health and wellness. "
        "You should only provide responses related to medical advice, wellness, mental health, fitness, "
        "nutrition, and related topics. If the user asks something outside health, politely refuse."
    )

    # Retrieve conversation history
    history = memory.load_memory_variables({})["history"]

    # Format the prompt with memory
    full_prompt = system_prompt + "\n" + history + "\nUser: " + query

    # Get response from Gemini
    response = llm.invoke(full_prompt)

    # Save conversation to memory
    memory.save_context({"input": str(query)}, {"output": str(response)})

    return response


st.title("🩺 Health & Wellness Chatbot")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# User Input
question = st.chat_input("Ask me about health and wellness...")
if question:
    # Display user message
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    # Get response from AI
    response = health_care_chatbot(question)

    # Display AI response
    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.write(response.content)
