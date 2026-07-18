"""
ChatAgent abstraction layer.

This module defines the ChatAgent class, which serves as a decoupled abstraction
between the display interface (CLI/UI) and the underlying agent logic.
"""
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass

@dataclass
class ChatEvent:
    """Base class for all events in the chat history."""
    event_type: str

@dataclass
class MessageEvent(ChatEvent):
    role: str  # 'user' or 'agent'
    content: str

@dataclass
class MessageChunkEvent(ChatEvent):
    """Fired when a chunk of a message is received (for streaming)."""
    role: str
    chunk: str

@dataclass
class ToolRequestEvent(ChatEvent):
    tool_name: str
    arguments: Dict[str, Any]

@dataclass
class ToolResultEvent(ChatEvent):
    tool_name: str
    result: Any

class HumanInTheLoopEvent(ChatEvent):
    pass

class ChatAgent:
    """
    A unified abstraction for an agentic chatbot session.
    
    This class uses an event-driven architecture. UIs can subscribe to events
    (like MessageChunkEvent for streaming) and update themselves accordingly.
    """
    
    def __init__(self, session_id: Optional[str] = None):
        """
        Initialize a new or existing chat session.
        """
        self.history: List[ChatEvent] = []
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
        pass
        
    def get_history(self) -> List[ChatEvent]:
        """
        Retrieve the current conversation history.
        """
        return self.history

if __name__ == "__main__":
    agent = ChatAgent()
    
    # 1. Define a UI listener function
    def my_ui_renderer(event: ChatEvent):
        if isinstance(event, MessageChunkEvent):
            # Stream tokens directly to the console
            print(event.chunk, end="", flush=True)
            
        elif isinstance(event, MessageEvent) and event.role == "agent":
            print(f"\n[Agent Finished]: {event.content}")
            
        elif isinstance(event, ToolRequestEvent):
            print(f"\n[Agent used tool]: {event.tool_name} with {event.arguments}")
            
    # 2. Subscribe the UI to the agent
    agent.add_listener(my_ui_renderer)
    
    # 3. Send message (agent will now emit events to the renderer)
    agent.send_message("Hello, how are you?")