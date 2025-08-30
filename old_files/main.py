from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from dotenv import load_dotenv
import os
import time
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory

# Load environment variables from the config file
load_dotenv("Environments/environment_tut1.env")

langsmith_api_key = os.getenv("LANGSMITH_API_KEY")
gemini_api_key = os.getenv("gemini_API_KEY")

# Initialize conversation memory
memory = ConversationBufferMemory(k=15)

# Initialize the LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=gemini_api_key)

def sexual_health_chatbot(query: str) -> str:
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
    full_prompt = f"{system_prompt}\n{history}\nUser: {query}"

    # Get response from Gemini
    response = llm.invoke(full_prompt)

    # Save conversation to memory
    memory.save_context({"input": query}, {"output": str(response)})

    # Return the response content (or the string representation)
    try:
        return response.content
    except AttributeError:
        return str(response)

app = FastAPI()

@app.get("/", response_class=FileResponse)
def get_home():
    # Serve the separate HTML file
    return FileResponse("index.html")

@app.post("/chat")
async def chat_endpoint(payload: dict):
    question = payload.get("question")
    if not question:
        return JSONResponse({"error": "No question provided"}, status_code=400)
    
    answer = sexual_health_chatbot(question)
    return {"response": answer}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
