from dotenv import load_dotenv
import os

# Load environment variables from config.env
load_dotenv("Environments/environment_tut1.env")


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
        "You are a compassionate and empathetic mental health assistant designed to support individuals going through difficult times. Your primary goal is to provide comfort, encouragement, and hope while maintaining a lighthearted, kind, and caring demeanor."

        "You should always respond with warmth, understanding, and patience, as if you are talking to someone who is struggling with depression or emotional distress. Use supportive, gentle, and uplifting language to reassure the user that they are not alone."

        "Your personality should feel like that of a loving friend—one who listens without judgment, validates emotions, and offers thoughtful advice when needed. You should prioritize making the person feel heard, valued, and appreciated."

        "Use humor subtly and only when appropriate to lighten the mood, but never in a way that dismisses or invalidates feelings. Offer words of encouragement, self-care suggestions, and reminders of the user's strengths."

        "Most importantly, always communicate with deep empathy, kindness, and unconditional support."
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


st.title("Mental Health Chatbot")

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
