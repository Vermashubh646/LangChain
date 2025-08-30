from dotenv import load_dotenv
import os
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
import time

# Load environment variables from config.env
load_dotenv("Environments/environment_tut1.env")

langsmith_api_key = os.getenv("LANGSMITH_API_KEY")
gemini_api_key = os.getenv("gemini_API_KEY")

# Configure page to be more private
st.set_page_config(
    page_title="Private Health Consultation",
    page_icon="🔒",
    initial_sidebar_state="collapsed"
)

# Initialize memory to store conversation
memory = ConversationBufferMemory(k=15)

def sexual_health_chatbot(query):
    system_prompt = (
        "You are a compassionate sexual health educator and counselor. Your role is to provide accurate, non-judgmental information about sexual health, STIs, reproductive health, and related topics. "
        "Explain medical concepts in clear, accessible language while maintaining medical accuracy. "
        "Be sensitive and respectful when discussing intimate topics. Provide evidence-based information "
        "and emphasize the importance of consulting healthcare professionals for diagnosis and treatment. "
        "Focus on facts rather than moral judgments. Maintain a professional, supportive tone throughout. "
        "When appropriate, mention resources for testing, treatment, or support services without providing specific locations. "
        "Always prioritize user privacy and confidentiality in your responses."
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

# Initialize LLM
@st.cache_resource
def initialize_llm():
    return ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=gemini_api_key)

llm = initialize_llm()

# Privacy notice and features
with st.sidebar:
    st.title("Privacy Features")
    st.info(
        "🔒 This chat is designed for private sexual health discussions. Your conversations "
        "are not permanently stored on our servers and will be cleared when you close the browser "
        "or click 'New Session' below."
    )
    
    if st.button("New Session 🗑️"):
        # Clear memory and chat history
        memory.clear()
        st.session_state.messages = []
        st.session_state.show_privacy_reminder = False
        st.experimental_rerun()
    
    auto_clear = st.checkbox("Auto-clear after inactivity (5 min)", value=True)
    
    st.markdown("---")
    st.caption("Remember: For medical emergencies or diagnosis, please consult a healthcare professional.")

# Main interface
st.title("Private Sexual Health Consultation")
st.markdown(
    """
    Ask questions about sexual health, STIs, reproductive health, and related topics.
    This is a private space designed for confidential discussions.
    """
)

# Privacy reminder
if "show_privacy_reminder" not in st.session_state:
    st.session_state.show_privacy_reminder = True
    st.session_state.last_interaction_time = time.time()

if st.session_state.show_privacy_reminder:
    with st.expander("Privacy Information (click to expand)", expanded=True):
        st.warning(
            "This is a private consultation space. However, please avoid sharing personal "
            "identifiable information. For your privacy:"
            "\n- No login required"
            "\n- Your conversation will be cleared when you close this page"
            "\n- You can start a new session anytime using the sidebar"
        )
        if st.button("I understand"):
            st.session_state.show_privacy_reminder = False
            st.experimental_rerun()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Auto-clear functionality
if "last_interaction_time" in st.session_state and auto_clear:
    inactive_time = time.time() - st.session_state.last_interaction_time
    if inactive_time > 300:  # 5 minutes in seconds
        memory.clear()
        st.session_state.messages = []
        st.info("Session cleared due to inactivity")
        st.session_state.last_interaction_time = time.time()

# User Input
question = st.chat_input("Ask your question about sexual health...")
if question:
    # Update last interaction time
    st.session_state.last_interaction_time = time.time()
    
    # Display user message
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    # Show typing indicator
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.text("Thinking...")
        
        # Get response from AI
        response = sexual_health_chatbot(question)
        
        # Display AI response
        message_placeholder.write(response.content)
    
    # Add to session state
    st.session_state.messages.append({"role": "assistant", "content": response.content})