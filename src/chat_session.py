import uuid
import json
from typing import Iterator, Dict, Any, List

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, AIMessageChunk

class ChatSession:
    def __init__(self, agent, thread_id: str = None):
        self.agent = agent
        self.thread_id = thread_id or str(uuid.uuid4())
        self.config = {"configurable": {"thread_id": self.thread_id}}

    def get_history(self) -> List[Dict[str, Any]]:
        state = self.agent.get_state(self.config)
        messages = state.values.get("messages", []) if state and hasattr(state, 'values') and state.values else []
        
        history = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                history.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                content_str = ""
                if msg.content:
                    if isinstance(msg.content, str):
                        content_str = msg.content
                    elif isinstance(msg.content, list):
                        for block in msg.content:
                            if isinstance(block, dict) and block.get("type") == "text" and "text" in block:
                                content_str += block["text"]
                
                tool_calls = []
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    for tc in msg.tool_calls:
                        tool_calls.append({
                            "name": tc.get('name', ''),
                            "args": tc.get('args', {}),
                            "id": tc.get('id', '')
                        })
                
                history.append({
                    "role": "assistant",
                    "content": content_str,
                    "tool_calls": tool_calls
                })
            elif isinstance(msg, ToolMessage):
                history.append({
                    "role": "tool", 
                    "name": msg.name,
                    "result": msg.content,
                    "id": getattr(msg, "tool_call_id", "")
                })
                
        return history

    def send_message(self, prompt: str) -> Iterator[Dict[str, Any]]:
        input_state = {"messages": [HumanMessage(content=prompt)]}
        active_tool_calls = {}

        for chunk in self.agent.stream(input_state, self.config, stream_mode="messages", version="v2"):
            if chunk["type"] == "messages":
                msg, metadata = chunk["data"]

                if isinstance(msg, (AIMessage, AIMessageChunk)):
                    if msg.content:
                        content_str = ""
                        if isinstance(msg.content, str):
                            content_str = msg.content
                        elif isinstance(msg.content, list):
                            for block in msg.content:
                                if isinstance(block, dict) and block.get("type") == "text" and "text" in block:
                                    content_str += block["text"]
                        
                        if content_str:
                            yield {"type": "content_chunk", "data": content_str}

                    if hasattr(msg, "tool_call_chunks") and msg.tool_call_chunks:
                        for tc in msg.tool_call_chunks:
                            idx = tc.get("index")
                            if idx not in active_tool_calls:
                                active_tool_calls[idx] = {"name": tc.get("name", ""), "args": "", "id": tc.get("id", "")}
                            if tc.get("args"):
                                active_tool_calls[idx]["args"] += tc["args"]

                elif isinstance(msg, ToolMessage):
                    # Find matching tool call to pass arguments
                    tc_args = "{}"
                    for tc in active_tool_calls.values():
                        if tc["id"] == getattr(msg, "tool_call_id", "") or tc["name"] == msg.name:
                            tc_args = tc["args"]
                            break
                    
                    try:
                        parsed_args = json.loads(tc_args) if tc_args else {}
                        args_str = json.dumps(parsed_args, indent=2)
                    except json.JSONDecodeError:
                        args_str = tc_args
                        
                    yield {
                        "type": "tool_execution",
                        "name": msg.name,
                        "args": args_str,
                        "result": msg.content
                    }
