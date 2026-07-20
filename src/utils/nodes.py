import os
from langchain_core.messages import SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_deepseek import ChatDeepSeek
from .state import AgentState
from .tools import tools

def get_llm(provider: str):
    if provider == 'deepseek':
        return ChatDeepSeek(
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
    elif provider == 'google':
        return ChatGoogleGenerativeAI(
            model=os.environ.get('GEMINI_MODEL'),
            thinking_level='low',
            temperature=0,
            max_retries=2,
        )
    else:
        raise ValueError(f"Unknown provider: {provider}")

llm = get_llm('google') # By default use Google; if encounter 20 RPM limit, change to Deepseek
llm_with_tools = llm.bind_tools(tools)

system_prompt = """You are a helpful AI assistant connected to a local SQLite database sandbox.
You have access to tools to query the database.
Be brief in your responses."""

def call_model(state: AgentState):
    # Prepend the system prompt safely for this invocation
    messages = [SystemMessage(content=system_prompt)] + state.messages
    
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}
