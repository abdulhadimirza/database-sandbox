import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

st.title("Database Sandbox")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Function to generate streaming response from Gemini
def stream_gemini_response(chat_history):
    # Convert history to LangChain messages format
    formatted_messages = []
    for msg in chat_history:
        if msg["role"] == "user":
            formatted_messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            formatted_messages.append(AIMessage(content=msg["content"]))

    try:
        # Initialize the Gemini model
        llm = ChatGoogleGenerativeAI(model="gemini-flash-lite-latest")
        for chunk in llm.stream(formatted_messages):
            for part in chunk.content:
                yield part["text"]
    except Exception as e:
        yield f"Error calling Gemini API: {str(e)}"

# Handle new user input
if prompt := st.chat_input("Say something"):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.write(prompt)
    
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Generate and display assistant response using streaming
    with st.chat_message("assistant"):
        response = st.write_stream(stream_gemini_response(st.session_state.messages))
            
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
