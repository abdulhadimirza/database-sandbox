import streamlit as st
import os
import uuid
from dotenv import load_dotenv

load_dotenv()

from agent import agent
from langchain_core.messages import HumanMessage

st.title("Database Sandbox")

# Initialize thread_id for LangGraph MemorySaver
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Function to generate streaming response from the agent
def stream_gemini_response(prompt_message):
    try:
        config = {"configurable": {"thread_id": st.session_state.thread_id}}
        
        # Only pass the new human message to the agent, memory handles the rest
        input_state = {"messages": [HumanMessage(content=prompt_message)]}
        
        # Stream response from the LangGraph agent
        for chunk in agent.stream(input_state, config, stream_mode="messages", version="v2"):
            if chunk["type"] == "messages":
                msg, metadata = chunk["data"]
                # Only yield text content from the main agent node (ignore tool executions)
                if msg.content and metadata.get("langgraph_node") != "tools":
                    if isinstance(msg.content, str):
                        yield msg.content
                    elif isinstance(msg.content, list):
                        for block in msg.content:
                            if isinstance(block, dict) and block.get("type") == "text" and "text" in block:
                                yield block["text"]
    except Exception as e:
        yield f"Error calling agent: {str(e)}"

# Handle new user input
if prompt := st.chat_input("Say something"):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.write(prompt)
    
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Generate and display assistant response using streaming
    with st.chat_message("assistant"):
        response = st.write_stream(stream_gemini_response(prompt))
            
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
