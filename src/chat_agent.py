"""
ChatAgent abstraction layer.

This module defines the ChatAgent class, which serves as a decoupled abstraction
between the display interface (CLI/UI) and the underlying agent logic.
"""
from dotenv import load_dotenv
load_dotenv()

import warnings
from langchain_core._api import LangChainBetaWarning
warnings.filterwarnings("ignore", category=LangChainBetaWarning)

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
class UserMessageEvent(ChatEvent):
    content: str
    event_type: str = field(default="user_message", init=False)

@dataclass
class AgentMessageCompleteEvent(ChatEvent):
    content: str
    event_type: str = field(default="agent_message_complete", init=False)

@dataclass
class AgentThinkingEvent(ChatEvent):
    event_type: str = field(default="agent_thinking", init=False)

@dataclass
class AgentToolRequestEvent(ChatEvent):
    tool_name: str
    arguments: Dict[str, Any]
    event_type: str = field(default="agent_tool_request", init=False)

@dataclass
class AgentToolResultEvent(ChatEvent):
    tool_name: str
    arguments: Dict[str, Any]
    result: Any
    event_type: str = field(default="agent_tool_result", init=False)

@dataclass
class AgentToolErrorEvent(ChatEvent):
    tool_name: str
    arguments: Dict[str, Any]
    error: str
    event_type: str = field(default="agent_tool_error", init=False)

@dataclass
class AgentMessageStartEvent(ChatEvent):
    event_type: str = field(default="agent_message_start", init=False)

@dataclass
class AgentMessageChunkEvent(ChatEvent):
    chunk: str
    event_type: str = field(default="agent_message_chunk", init=False)

@dataclass
class AgentTurnCompleteEvent(ChatEvent):
    event_type: str = field(default="agent_turn_complete", init=False)

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
        if not isinstance(event, (AgentMessageChunkEvent, AgentThinkingEvent)):
            self.history.append(event)
            
        for listener in self._listeners:
            listener(event)
            
    def send_message(self, message: str) -> None:
        """
        Initiate sending a message. The agent's progress and response 
        will be communicated entirely via event listeners.
        """
        self._emit(UserMessageEvent(content=message))
        
        input_state = {"messages": [{"role": "user", "content": message}]}

        stream = agent.stream_events(
            input_state,
            self.config,
            version="v3",
            transformers=[ToolCallTransformer]
        )

        active_tools: Dict[str, Dict[str, Any]] = {}
        current_msg_buffer = ""
        self._emit(AgentThinkingEvent())
        
        for event in stream:
            if event["method"] == "messages":
                payload_dict = event["params"]["data"][0]
                event_type = payload_dict["event"]

                if event_type == "content-block-start":
                    block_type = payload_dict["content"]["type"]
                    if block_type == "text":
                        self._emit(AgentMessageStartEvent())
                        current_msg_buffer = ""
                elif event_type == "content-block-delta":
                    delta = payload_dict["delta"]
                    if delta["type"] == "text-delta":
                        text_chunk = delta["text"]
                        self._emit(AgentMessageChunkEvent(chunk=text_chunk))
                        current_msg_buffer += text_chunk
                elif event_type == "content-block-finish":
                    if current_msg_buffer:
                        self._emit(AgentMessageCompleteEvent(content=current_msg_buffer))
                        current_msg_buffer = ""
            elif event["method"] == "tools":
                data = event["params"]["data"]
                if data["event"] == "tool-started":
                    tool_name = data["tool_name"]
                    tool_input = data["input"]
                    tool_call_id = data["tool_call_id"]
                    active_tools[tool_call_id] = {
                        "tool_name": tool_name,
                        "input": tool_input
                    }
                    self._emit(AgentToolRequestEvent(tool_name=tool_name, arguments=tool_input))
                elif data["event"] == "tool-finished":
                    tool_message = data["output"]
                    tool_output = tool_message.content if hasattr(tool_message, "content") else str(tool_message)
                    tool_call_id = data["tool_call_id"]
                    active_tool = active_tools.pop(tool_call_id, {})
                    t_name = active_tool.get("tool_name", "Unknown")
                    t_input = active_tool.get("input", {})
                    self._emit(AgentToolResultEvent(tool_name=t_name, arguments=t_input, result=tool_output))
                    self._emit(AgentThinkingEvent())
                elif data["event"] == "tool-error":
                    tool_call_id = data["tool_call_id"]
                    active_tool = active_tools.pop(tool_call_id, {})
                    t_name = active_tool.get("tool_name", "Unknown")
                    t_input = active_tool.get("input", {})
                    self._emit(AgentToolErrorEvent(tool_name=t_name, arguments=t_input, error="Tool Failed"))
                    self._emit(AgentThinkingEvent())
                    
        self._emit(AgentTurnCompleteEvent())
        
    def get_history(self) -> List[ChatEvent]:
        """
        Retrieve the current conversation history.
        """
        return self.history

if __name__ == "__main__":
    print("\n--- Initializing Agent ---")
    testAgent = ChatAgent()
    
    # 1. Define a UI listener function
    def my_ui_renderer(event: ChatEvent):
        if isinstance(event, AgentMessageChunkEvent):
            # Print chunks on the same line to test streaming
            print(event.chunk, end="", flush=True)
        else:
            # Print other events with their type to clearly see the order
            print(f"\n[EVENT EMITTED] {type(event).__name__}")
            if isinstance(event, AgentMessageCompleteEvent):
                print(f"   Content Length: {len(event.content)} characters")
            elif isinstance(event, UserMessageEvent):
                print(f"   User Says: {event.content}")
            elif isinstance(event, AgentToolRequestEvent):
                print(f"   Tool: {event.tool_name} requested")
            elif isinstance(event, AgentToolResultEvent):
                print(f"   Tool: {event.tool_name} returned result")
            
    # 2. Subscribe the UI to the agent
    testAgent.add_listener(my_ui_renderer)
    
    # 3. Send message (agent will now emit events to the renderer)
    print("\n--- Sending First Message ---")
    testAgent.send_message("List the tables in the database and describe them.")
    
    # 4. Verify History Order
    print("\n\n--- Verifying History Order ---")
    history = testAgent.get_history()
    for i, event in enumerate(history):
        print(f"History[{i}]: {type(event).__name__}")