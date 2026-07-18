import streamlit as st
import os
import uuid
import json
from dotenv import load_dotenv

load_dotenv()

from agent import agent
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, AIMessageChunk

st.title("Database Sandbox")

# Initialize thread_id for LangGraph MemorySaver
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "current_key" not in st.session_state:
    st.session_state.current_key = -1

# Fetch chat history from LangGraph checkpointer
config = {"configurable": {"thread_id": st.session_state.thread_id}}
state = agent.get_state(config)
messages = state.values.get("messages", []) if state and hasattr(state, 'values') and state.values else []

# Group contiguous messages by role to prevent separate chat bubbles for tool calls
grouped_messages = []
for msg in messages:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    if not grouped_messages or grouped_messages[-1]["role"] != role:
        grouped_messages.append({"role": role, "messages": [msg]})
    else:
        grouped_messages[-1]["messages"].append(msg)

# Display chat history from LangGraph state
history_placeholder = st.empty()
with history_placeholder.container():
    for group in grouped_messages:
        with st.chat_message(group["role"]):
            for msg in group["messages"]:
                if isinstance(msg, HumanMessage):
                    st.markdown(msg.content)
                elif isinstance(msg, AIMessage):
                    if msg.content:
                        if isinstance(msg.content, str):
                            st.markdown(msg.content)
                        elif isinstance(msg.content, list):
                            for block in msg.content:
                                if isinstance(block, dict) and block.get("type") == "text" and "text" in block:
                                    st.markdown(block["text"])
                    if hasattr(msg, "tool_calls") and msg.tool_calls:
                        for tc in msg.tool_calls:
                            with st.status(f"↻ Tool Call: {tc['name']}", state="complete"):
                                st.write(tc['args'])
                elif isinstance(msg, ToolMessage):
                    with st.status(f"✓ Tool Result: {msg.name}", state="complete"):
                        st.code(msg.content)

# Handle new user input
prompt = st.chat_input("Say something")
if prompt:
    input_state = {"messages": [HumanMessage(content=prompt)]}
    
    # Wrap in st.empty() to completely clear any stale grey elements from the previous stream rerun
    user_placeholder = st.empty()
    with user_placeholder.container():
        with st.chat_message("user"):
            st.markdown(prompt)
    
    # Create an explicit rendering zone right before streaming the response
    assist_placeholder = st.empty()
    with assist_placeholder.container():
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            active_tool_calls = {}
            
            # Stream response from the LangGraph agent
            for chunk in agent.stream(input_state, config, stream_mode="messages", version="v2"):
                if chunk["type"] == "messages":
                    msg, metadata = chunk["data"]
                    
                    if isinstance(msg, (AIMessage, AIMessageChunk)):
                        if msg.content:
                            if isinstance(msg.content, str):
                                full_response += msg.content
                            elif isinstance(msg.content, list):
                                for block in msg.content:
                                    if isinstance(block, dict) and block.get("type") == "text" and "text" in block:
                                        full_response += block["text"]
                            message_placeholder.markdown(full_response + "▌")
                            
                        if hasattr(msg, "tool_call_chunks") and msg.tool_call_chunks:
                            for tc in msg.tool_call_chunks:
                                idx = tc.get("index")
                                if idx not in active_tool_calls:
                                    active_tool_calls[idx] = {"name": tc.get("name", ""), "args": "", "id": tc.get("id", "")}
                                if tc.get("args"):
                                    active_tool_calls[idx]["args"] += tc["args"]
                            
                    elif isinstance(msg, ToolMessage):
                        # Find matching tool call to display arguments
                        tc_args = "{}"
                        for tc in active_tool_calls.values():
                            if tc["id"] == getattr(msg, "tool_call_id", "") or tc["name"] == msg.name:
                                tc_args = tc["args"]
                                break
                        
                        try:
                            parsed_args = json.loads(tc_args) if tc_args else {}
                        except json.JSONDecodeError:
                            parsed_args = tc_args
                            
                        with st.status(f"↻ Tool Call: {msg.name}", state="complete"):
                            st.write(parsed_args)
                            
                        with st.status(f"✓ Tool Result: {msg.name}", state="complete"):
                            st.code(msg.content)
                        
                        # Reset placeholder for any subsequent AI message
                        message_placeholder = st.empty()
                        full_response = ""
                        
            message_placeholder.markdown(full_response)
