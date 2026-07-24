import os
import time

from langchain_core.messages import SystemMessage
from langchain_deepseek import ChatDeepSeek
from langchain_google_genai import ChatGoogleGenerativeAI

from .state import AgentState

deepseek = ChatDeepSeek(
    api_key=os.environ.get('DEEPSEEK_API_KEY') or 'dummy-key',
    model=os.environ.get('DEEPSEEK_MODEL') or 'deepseek-v4-flash',
    reasoning_effort='low',
    temperature=1.0,
    max_retries=2,
    extra_body={
        'thinking': {
            'type': 'disabled'
        }
    },
)

gemini = ChatGoogleGenerativeAI(
    api_key=os.environ.get('GEMINI_API_KEY') or 'dummy-key',
    model=os.environ.get('GEMINI_MODEL') or 'gemini-flash-lite-latest',
    thinking_level='low',
    temperature=1.0,
    max_retries=2,
)



_rate_limit_reset_time = 0.0

def create_agent_node(system_prompt: str, node_tools: list):
    # Bind tools specific to this agent
    primary = gemini.bind_tools(node_tools)
    fallback = deepseek.bind_tools(node_tools)
    
    # Define the actual LangGraph node function
    def node(state: AgentState):
        global _rate_limit_reset_time
        
        # Keep recent message context to avoid exceeding model token limits
        #recent_messages = state.messages[-25:] if len(state.messages) > 25 else state.messages
        #messages = [SystemMessage(content=system_prompt)] + recent_messages
        messages = [SystemMessage(content=system_prompt)] + state.messages

        current_time = time.time()
        
        if current_time < _rate_limit_reset_time:
            response = fallback.invoke(messages)
        else:
            try:
                response = primary.invoke(messages)
            except Exception as e:
                # langchain-google-genai wraps the APIError in a ChatGoogleGenerativeAIError.
                # The original APIError is preserved in e.__cause__.
                cause = getattr(e, "__cause__", None)
                if (
                    (cause and getattr(cause, "code", None) == 429) or 
                    (cause and getattr(cause, "status", None) == 429) or 
                    "429" in str(e)
                ):
                    _rate_limit_reset_time = time.time() + 60.0
                    response = fallback.invoke(messages)
                else:
                    raise
                    
        return {"messages": [response]}
        
    return node

