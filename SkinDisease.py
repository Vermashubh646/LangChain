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

# List of skin diseases with their prevalence
skin_diseases = {
    "Eczema": "A chronic condition causing inflamed, itchy, and cracked skin.",
    "Melanoma": "A serious form of skin cancer that develops in melanocytes.",
    "Atopic Dermatitis": "A type of eczema that makes skin red and itchy.",
    "Basal Cell Carcinoma (BCC)": "A common skin cancer that arises from basal cells.",
    "Melanocytic Nevi (NV)": "Commonly known as moles, these are benign skin growths.",
    "Benign Keratosis-like Lesions (BKL)": "Non-cancerous skin growths often seen in aging skin.",
    "Psoriasis, Lichen Planus, and related diseases": "Autoimmune conditions causing red, scaly patches on the skin.",
    "Seborrheic Keratoses and other Benign Tumors": "Non-cancerous, wart-like skin growths.",
    "Tinea, Ringworm, Candidiasis, and other Fungal Infections": "Fungal infections affecting the skin, hair, and nails.",
    "Warts, Molluscum, and other Viral Infections": "Skin conditions caused by viral infections."
}

def skin_disease_chatbot(query):
    system_prompt = (
        "You are a knowledgeable dermatology assistant specialized in identifying and providing information about skin diseases."
        "Your role is to educate users about symptoms, causes, prevalence, and treatments of common skin conditions in a clear and concise manner."
    )
    
    # Retrieve conversation history
    history = memory.load_memory_variables({})["history"]
    
    # Check if the query matches a known skin disease
    response = "I'm here to help! Could you specify which skin condition you'd like to learn about?"
    for disease, info in skin_diseases.items():
        if disease.lower() in query.lower():
            response = f"{disease}: {info}"
            break
    
    # Save conversation to memory
    memory.save_context({"input": str(query)}, {"output": str(response)})
    
    return response

st.title("Skin Disease Chatbot")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# User Input
question = st.chat_input("Ask me about skin diseases...")
if question:
    # Display user message
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    # Get response from AI
    response = skin_disease_chatbot(question)

    # Display AI response
    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.write(response)
