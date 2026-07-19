"""
ChatAgent abstraction layer.

This module defines the ChatAgent class, which serves as a decoupled abstraction
between the display interface (CLI/UI) and the underlying agent logic.
"""
from google.genai.types import ToolCallDict
from dotenv import load_dotenv
load_dotenv()

from uuid import uuid4
from dataclasses import dataclass, field
from typing import List, Dict, Any, Callable
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.prebuilt import ToolCallTransformer

from agent import agent

@dataclass
class ChatEvent:
    """Base class for all events in the chat history."""
    event_type: str = field(init=False)

@dataclass
class MessageEvent(ChatEvent):
    role: str  # 'user' or 'agent'
    content: str
    event_type: str = field(default="message", init=False)

@dataclass
class MessageChunkEvent(ChatEvent):
    """Fired when a chunk of a message is received (for streaming)."""
    role: str
    chunk: str
    event_type: str = field(default="message_chunk", init=False)

@dataclass
class ToolRequestEvent(ChatEvent):
    tool_name: str
    arguments: Dict[str, Any]
    event_type: str = field(default="tool_request", init=False)

@dataclass
class ToolResultEvent(ChatEvent):
    tool_name: str
    result: Any
    event_type: str = field(default="tool_result", init=False)

class HumanInTheLoopEvent(ChatEvent):
    event_type: str = field(default="human_in_the_loop", init=False)

class ChatAgent:
    """
    A unified abstraction for an agentic chatbot session.
    
    This class uses an event-driven architecture. UIs can subscribe to events
    (like MessageChunkEvent for streaming) and update themselves accordingly.
    """
    
    def __init__(self):
        """
        Initialize a new or existing chat session.
        """
        self.history: List[ChatEvent] = []
        self.config = {"configurable": {"thread_id": str(uuid4())}, "recursion_limit": 10}
        self._listeners: List[Callable[[ChatEvent], None]] = []
    
    def add_listener(self, listener: Callable[[ChatEvent], None]) -> None:
        """
        Subscribe a callback function to listen to all chat events.
        """
        self._listeners.append(listener)
        
    def _emit(self, event: ChatEvent) -> None:
        """
        Internal method to add an event to history and notify listeners.
        Note: Chunk events are usually just emitted, while full messages are saved to history.
        """
        if not isinstance(event, MessageChunkEvent):
            self.history.append(event)
            
        for listener in self._listeners:
            listener(event)
            
    def send_message(self, message: str) -> None:
        """
        Initiate sending a message. The agent's progress and response 
        will be communicated entirely via event listeners.
        """
        
        input_state = {"messages": [{"role": "user", "content": message}]}

        stream = agent.stream_events(
            input_state,
            self.config,
            version="v3",
            transformers=[ToolCallTransformer]
        )

        active_tools = {}
        for event in stream:
            if event["method"] == "messages":
                #print("\n[DEBUG]\n", event)
                payload_dict = event["params"]["data"][0]
                # metadata_dict = event["params"]["data"][1]

                event_type = payload_dict["event"]

                if event_type == "content-block-delta":
                    delta = payload_dict["delta"]
                    if delta["type"] == "text-delta":
                        print(delta["text"], end="", flush=True)
                elif event_type == "message-start":
                    role = payload_dict["role"]
                    print(f"\n[Message Start] Role: {role}")
                elif event_type == "message-finish":
                    print(f"\n[Message Finish]")
                elif event_type == "content-block-start":
                    block_type = payload_dict["content"]["type"]
                    print(f"\n[Content Block Start] Type: {block_type}")
                elif event_type == "content-block-finish":
                    block_type = payload_dict["content"]["type"]
                    print(f"\n[Content Block Finish] Type: {block_type}")
            elif event["method"] == "tools":
                #print("\n[DEBUG]\n", event)
                data = event["params"]["data"]
                if data["event"] == "tool-started":
                    tool_name = data["tool_name"]
                    tool_input = data["input"]
                    tool_call_id = data["tool_call_id"]
                    active_tools[tool_call_id] = {
                        "tool_name": tool_name,
                        "input": tool_input
                    }
                    print(f"Tool Request: {tool_name} with input {tool_input}")
                elif data["event"] == "tool-finished":
                    tool_message = data["output"]
                    tool_output = tool_message.content if hasattr(tool_message, "content") else str(tool_message)
                    tool_call_id = data["tool_call_id"]
                    active_tool = active_tools.pop(tool_call_id, {})
                    t_name = active_tool.get("tool_name", "Unknown")
                    t_input = active_tool.get("input", {})
                    print(f"Tool Finished [{tool_call_id}] - {t_name}({t_input}) -> Result: {tool_output}")
                elif data["event"] == "tool-error":
                    tool_call_id = data["tool_call_id"]
                    active_tool = active_tools.pop(tool_call_id, {})
                    t_name = active_tool.get("tool_name", "Unknown")
                    t_input = active_tool.get("input", {})
                    print(f"Tool Failed [{tool_call_id}] - {t_name}({t_input})")
        
    def get_history(self) -> List[ChatEvent]:
        """
        Retrieve the current conversation history.
        """
        return self.history

if __name__ == "__main__":
    testAgent = ChatAgent()
    
    # 1. Define a UI listener function
    def my_ui_renderer(event: ChatEvent):
        print("[UI RENDERER]: ", event)
            
    # 2. Subscribe the UI to the agent
    testAgent.add_listener(my_ui_renderer)
    
    # 3. Send message (agent will now emit events to the renderer)
    testAgent.send_message("List the tables in the database and describe them.")