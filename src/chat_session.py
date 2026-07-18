import uuid
import json
from typing import Iterator, Dict, Any, List, TypedDict, Union, Literal, Optional

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, AIMessageChunk
from langgraph.graph.state import CompiledStateGraph

class HistoryUserMessage(TypedDict):
    role: Literal["user"]
    content: str

class HistoryAssistantMessage(TypedDict):
    role: Literal["assistant"]
    content: str

class HistoryToolMessage(TypedDict):
    role: Literal["tool"]
    name: str
    args: Dict[str, Any]
    result: str

HistoryMessage = Union[HistoryUserMessage, HistoryAssistantMessage, HistoryToolMessage]

class StreamContentEvent(TypedDict):
    type: Literal["content"]
    content: str

class StreamToolEvent(TypedDict):
    type: Literal["tool"]
    name: str
    args: Dict[str, Any]
    result: str
    id: str

StreamEvent = Union[StreamContentEvent, StreamToolEvent]

class ChatSession:
    def __init__(self, agent: CompiledStateGraph, thread_id: Optional[str] = None):
        self.agent = agent
        self.thread_id = thread_id or str(uuid.uuid4())
        self.config = {"configurable": {"thread_id": self.thread_id}}

    def get_history(self) -> List[HistoryMessage]:
        state = self.agent.get_state(self.config)
        messages = state.values.get("messages", []) if state and hasattr(state, 'values') and state.values else []
        
        history = []
        pending_tool_calls = {}

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
                
                if content_str:
                    history.append({
                        "role": "assistant",
                        "content": content_str
                    })

                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    for tc in msg.tool_calls:
                        pending_tool_calls[tc.get('id', '')] = tc
                        
            elif isinstance(msg, ToolMessage):
                tc_id = getattr(msg, "tool_call_id", "")
                tc = pending_tool_calls.get(tc_id, {})
                tc_args = tc.get("args", {})
                
                history.append({
                    "role": "tool", 
                    "name": msg.name,
                    "args": tc_args,
                    "result": msg.content
                })
                
        return history

    def send_message(self, prompt: str) -> Iterator[StreamEvent]:
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
                            yield {"type": "content", "content": content_str}

                    if hasattr(msg, "tool_call_chunks") and msg.tool_call_chunks:
                        for tc in msg.tool_call_chunks:
                            idx = tc.get("index")
                            if idx not in active_tool_calls:
                                active_tool_calls[idx] = {
                                    "name": tc.get("name") or "", 
                                    "args": "", 
                                    "id": tc.get("id") or ""
                                }
                            if tc.get("name"):
                                active_tool_calls[idx]["name"] = tc["name"]
                            if tc.get("id"):
                                active_tool_calls[idx]["id"] = tc["id"]
                            if tc.get("args"):
                                active_tool_calls[idx]["args"] += tc["args"]
                    elif hasattr(msg, "tool_calls") and msg.tool_calls:
                        for tc in msg.tool_calls:
                            idx = tc.get("id")
                            if idx not in active_tool_calls:
                                active_tool_calls[idx] = {
                                    "name": tc.get("name") or "",
                                    "args": json.dumps(tc.get("args") or {}),
                                    "id": tc.get("id") or ""
                                }

                elif isinstance(msg, ToolMessage):
                    # Find matching tool call to pass arguments
                    tc_args = "{}"
                    for tc_key, tc in list(active_tool_calls.items()):
                        if tc["id"] == getattr(msg, "tool_call_id", ""):
                            tc_args = tc["args"]
                            
                            # Delete the specific key from the dictionary
                            del active_tool_calls[tc_key]
                            break
                    
                    try:
                        parsed_args = json.loads(tc_args) if tc_args else {}
                    except json.JSONDecodeError:
                        parsed_args = {"_raw": tc_args}
                        
                    yield {
                        "type": "tool",
                        "name": msg.name,
                        "args": parsed_args,
                        "result": msg.content,
                    }
