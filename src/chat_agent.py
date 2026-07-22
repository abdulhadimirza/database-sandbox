"""
ChatAgent abstraction layer.

This module defines the ChatAgent class, which serves as a decoupled abstraction
between the display interface (CLI/UI) and the underlying agent logic.
"""
from dotenv import load_dotenv
load_dotenv()

import warnings
from langchain_core._api import LangChainBetaWarning
warnings.filterwarnings('ignore', category=LangChainBetaWarning)

from uuid import uuid4
from dataclasses import dataclass, field
from typing import List, Dict, Any, Callable
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, BaseMessage
from langgraph.prebuilt import ToolCallTransformer
from langgraph.errors import GraphRecursionError, GraphInterrupt
from langgraph.types import Command
from langchain_core.runnables import RunnableConfig

from agent import agent

@dataclass
class ChatEvent:
    """Base class for all events in the chat history."""
    event_type: str = field(init=False)

@dataclass
class UserMessageEvent(ChatEvent):
    content: str
    is_history: bool = False
    event_type: str = field(default='user_message', init=False)

@dataclass
class AgentMessageCompleteEvent(ChatEvent):
    content: str
    event_type: str = field(default='agent_message_complete', init=False)

@dataclass
class AgentThinkingEvent(ChatEvent):
    event_type: str = field(default='agent_thinking', init=False)

@dataclass
class AgentToolRequestEvent(ChatEvent):
    tool_name: str
    arguments: Dict[str, Any]
    event_type: str = field(default='agent_tool_request', init=False)

@dataclass
class AgentToolResultEvent(ChatEvent):
    tool_name: str
    arguments: Dict[str, Any]
    result: Any
    event_type: str = field(default='agent_tool_result', init=False)

@dataclass
class AgentToolErrorEvent(ChatEvent):
    tool_name: str
    arguments: Dict[str, Any]
    error: str
    event_type: str = field(default='agent_tool_error', init=False)

@dataclass
class AgentToolApprovalRequestEvent(ChatEvent):
    tool_name: str
    arguments: Dict[str, Any]
    message: str
    event_type: str = field(default='agent_tool_approval_request', init=False)

@dataclass
class AgentMessageStartEvent(ChatEvent):
    event_type: str = field(default='agent_message_start', init=False)

@dataclass
class AgentMessageChunkEvent(ChatEvent):
    chunk: str
    event_type: str = field(default='agent_message_chunk', init=False)

@dataclass
class AgentTurnCompleteEvent(ChatEvent):
    event_type: str = field(default='agent_turn_complete', init=False)

@dataclass
class AgentErrorEvent(ChatEvent):
    error: str
    event_type: str = field(default='agent_error', init=False)

class ChatAgent:
    """
    A unified abstraction for an agentic chatbot session.
    
    This class uses an event-driven architecture. UIs can subscribe to events
    (like MessageChunkEvent for streaming) and update themselves accordingly.
    """
    
    def __init__(self, thread_id: str = 'default_session'):
        """
        Initialize a new or existing chat session.
        """
        self.history: List[ChatEvent] = []
        self.config: RunnableConfig = {'configurable': {'thread_id': thread_id}, 'recursion_limit': 50}
        self._listeners: List[Callable[[ChatEvent], None]] = []
        self.is_paused: bool = False
        
    def load(self) -> None:
        """
        Load history and emit events to listeners.
        """
        self._restore_history()
        
    def _restore_history(self) -> None:
        """
        Restore chat history from the persisted state.
        """
        state = agent.get_state(self.config)
        if not state or 'messages' not in state.values:
            return
            
        messages: List[BaseMessage] = state.values.get('messages', [])
        tool_calls_map: Dict[str, Dict[str, Any]] = {}
        
        for msg in messages:
            if isinstance(msg, HumanMessage):
                self._emit(UserMessageEvent(content=msg.content, is_history=True))
            elif isinstance(msg, AIMessage):
                text_content = ""
                if isinstance(msg.content, str):
                    text_content = msg.content
                elif isinstance(msg.content, list):
                    for block in msg.content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            text_content += block.get("text", "")
                        elif isinstance(block, str):
                            text_content += block
                            
                if text_content:
                    self._emit(AgentMessageStartEvent())
                    self._emit(AgentMessageCompleteEvent(content=text_content))
                    
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    for tc in msg.tool_calls:
                        tool_calls_map[tc['id']] = tc
                        self._emit(AgentToolRequestEvent(
                            tool_name=tc['name'],
                            arguments=tc['args']
                        ))
                
                # If there are no tool calls, it means the agent finished its turn
                if not getattr(msg, 'tool_calls', []):
                    self._emit(AgentTurnCompleteEvent())
                    
            elif isinstance(msg, ToolMessage):
                tc = tool_calls_map.get(msg.tool_call_id)
                t_name = tc['name'] if tc else getattr(msg, 'name', 'Unknown')
                t_args = tc['args'] if tc else {}
                
                if getattr(msg, 'status', 'success') == 'error':
                    self._emit(AgentToolErrorEvent(
                        tool_name=t_name,
                        arguments=t_args,
                        error=msg.content if isinstance(msg.content, str) else str(msg.content)
                    ))
                else:
                    self._emit(AgentToolResultEvent(
                        tool_name=t_name,
                        arguments=t_args,
                        result=msg.content
                    ))
    
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
            try:
                listener(event)
            except Exception as e:
                print(f"[ChatAgent] Listener error: {e}")

    def _inject_error_to_agent_state(self, error_msg: str) -> None:
        """
        Inject an error message into the agent's state so the LLM is aware of the failure on the next turn.
        """
        agent.update_state(self.config, {"messages": [("system", f"The previous agent turn failed with error: {error_msg}")]})

    def _process_stream(self, stream) -> None:
        """
        Process the event stream returned by agent.stream_events.
        """
        self.is_paused = False
        active_tools: Dict[str, Dict[str, Any]] = {}
        current_msg_buffer = ''
        pending_tool_errors: List[AgentToolErrorEvent] = []
        
        try:
            self._emit(AgentThinkingEvent())
            
            for event in stream:
                if event['method'] == 'messages':
                    payload_dict = event['params']['data'][0]
                    event_type = payload_dict['event']

                    if event_type == 'content-block-start':
                        block_type = payload_dict['content']['type']
                        if block_type == 'text':
                            self._emit(AgentMessageStartEvent())
                            current_msg_buffer = ''
                    elif event_type == 'content-block-delta':
                        delta = payload_dict['delta']
                        if delta['type'] == 'text-delta':
                            text_chunk = delta['text']
                            self._emit(AgentMessageChunkEvent(chunk=text_chunk))
                            current_msg_buffer += text_chunk
                    elif event_type == 'content-block-finish':
                        if current_msg_buffer:
                            self._emit(AgentMessageCompleteEvent(content=current_msg_buffer))
                            current_msg_buffer = ''
                elif event['method'] == 'tools':
                    data = event['params']['data']
                    if data['event'] == 'tool-started':
                        tool_name = data['tool_name']
                        tool_input = data['input']
                        tool_call_id = data['tool_call_id']
                        active_tools[tool_call_id] = {
                            'tool_name': tool_name,
                            'input': tool_input
                        }
                        self._emit(AgentToolRequestEvent(tool_name=tool_name, arguments=tool_input))
                    elif data['event'] == 'tool-finished':
                        tool_message = data['output']
                        tool_output = tool_message.content if hasattr(tool_message, 'content') else str(tool_message)
                        tool_call_id = data['tool_call_id']
                        active_tool = active_tools.pop(tool_call_id, {})
                        t_name = active_tool.get('tool_name', 'Unknown')
                        t_input = active_tool.get('input', {})
                        
                        if getattr(tool_message, 'status', 'success') == 'error':
                            self._emit(AgentToolErrorEvent(tool_name=t_name, arguments=t_input, error=tool_output))
                        else:
                            self._emit(AgentToolResultEvent(tool_name=t_name, arguments=t_input, result=tool_output))
                            
                        self._emit(AgentThinkingEvent())
                    elif data['event'] == 'tool-error':
                        tool_call_id = data.get('tool_call_id')
                        active_tool = active_tools.pop(tool_call_id, {}) if tool_call_id else {}
                        t_name = active_tool.get('tool_name', 'Unknown')
                        t_input = active_tool.get('input', {})
                        err_msg = str(data.get('message') or data.get('error') or "Tool Failed")
                        pending_tool_errors.append(AgentToolErrorEvent(tool_name=t_name, arguments=t_input, error=err_msg))

            if getattr(stream, 'interrupted', False):
                self.is_paused = True
                # Stream was interrupted for human approval; discard transient tool errors
                pending_tool_errors.clear()
                interrupts = getattr(stream, 'interrupts', [])
                if interrupts:
                    payload = interrupts[0].value
                    if isinstance(payload, dict):
                        t_name = payload.get("action", "execute_write_query")
                        t_args = {"query": payload.get("query", "")}
                        t_msg = payload.get("message", "Approval required.")
                    else:
                        t_name = "execute_write_query"
                        t_args = {}
                        t_msg = str(payload)
                        
                    self._emit(AgentToolApprovalRequestEvent(
                        tool_name=t_name,
                        arguments=t_args,
                        message=t_msg
                    ))
            else:
                # Stream completed without interruption; emit any real pending tool errors
                for err_event in pending_tool_errors:
                    self._emit(err_event)
                    self._emit(AgentThinkingEvent())
        except GraphRecursionError as e:
            error_msg = f"Recursion limit reached: {str(e)}"
            self._emit(AgentErrorEvent(error=error_msg))
            self._inject_error_to_agent_state(error_msg)
        except Exception as e:
            error_msg = f"Agent execution failed: {str(e)}"
            self._emit(AgentErrorEvent(error=error_msg))
            self._inject_error_to_agent_state(error_msg)
        finally:
            self._emit(AgentTurnCompleteEvent())

    def send_message(self, message: str) -> None:
        """
        Initiate sending a message. The agent's progress and response 
        will be communicated entirely via event listeners.
        """
        self._emit(UserMessageEvent(content=message))
        
        input_state = {'messages': [{'role': 'user', 'content': message}]}
        
        stream = agent.stream_events(
            input_state,
            self.config,
            version='v3',
            transformers=[ToolCallTransformer]
        )
        self._process_stream(stream)
        
    def resume_turn(self, resume_data: Any) -> None:
        """
        Resume execution after an interrupt with the provided human feedback.
        """
        stream = agent.stream_events(
            Command(resume=resume_data),
            self.config,
            version='v3',
            transformers=[ToolCallTransformer]
        )
        self._process_stream(stream)
        
    def approve(self) -> None:
        """
        Approve the pending action/tool execution.
        """
        self.resume_turn({"action": "approve"})

    def reject(self) -> None:
        """
        Reject/cancel the pending action/tool execution.
        """
        self.resume_turn({"action": "reject"})

    def respond_to_approval(self, approved: bool) -> None:
        """
        Respond to a pending tool approval request with a boolean decision.
        """
        if approved:
            self.approve()
        else:
            self.reject()
        
    def get_history(self) -> List[ChatEvent]:
        """
        Retrieve the current conversation history.
        """
        return self.history

if __name__ == '__main__':
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
            elif isinstance(event, AgentErrorEvent):
                print(f"   Error: {event.error}")
            
    # 2. Subscribe the UI to the agent
    testAgent.add_listener(my_ui_renderer)
    testAgent.load()
    
    # 3. Send message (agent will now emit events to the renderer)
    print("\n--- Sending First Message ---")
    testAgent.send_message("List the tables in the database and describe them.")
    
    # 4. Verify History Order
    print("\n\n--- Verifying History Order ---")
    history = testAgent.get_history()
    for i, event in enumerate(history):
        print(f"History[{i}]: {type(event).__name__}")