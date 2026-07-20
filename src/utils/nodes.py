import os
import time

from langchain_core.messages import SystemMessage
from langchain_deepseek import ChatDeepSeek
from langchain_google_genai import ChatGoogleGenerativeAI

from .state import AgentState

deepseek = ChatDeepSeek(
    model=os.environ.get('DEEPSEEK_MODEL'),
    reasoning_effort='low',
    temperature=0,
    max_retries=2,
    extra_body={
        'thinking': {
            'type': 'disabled'
        }
    },
)

gemini = ChatGoogleGenerativeAI(
    model=os.environ.get('GEMINI_MODEL'),
    thinking_level='low',
    temperature=0,
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
