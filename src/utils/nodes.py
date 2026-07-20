from langchain_core.runnables import Runnable
from langchain_core.messages import AIMessage
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.tools import BaseTool
from typing import Callable
from typing import Sequence
from typing import override
from langgraph.checkpoint.base import BaseCheckpointSaver
import os
import time
from typing import Any, List, Optional
from pydantic import Field
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.outputs import ChatResult, ChatGeneration
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_deepseek import ChatDeepSeek
from google.genai.errors import APIError
from .state import AgentState
from .tools import tools
from langchain_core.runnables import RunnableConfig
from langchain_core.language_models import LanguageModelInput
from pydantic import Field, PrivateAttr


class RateLimitedFallbackChatModel(BaseChatModel):
    primary_model: Any = Field(description="The primary model (e.g., Gemini)")
    fallback_model: Any = Field(description="The fallback model (e.g., DeepSeek)")

    rate_limit_reset_time: float = 0

    @property
    def _llm_type(self) -> str:
        return "rate_limited_fallback"
    
    def _get_target_model(self) -> Any:
        current_time = time.time()
        
        # If we are within the cooldown window, use the fallback.
        if current_time < self.rate_limit_reset_time:
            return self.fallback_model
        else:
            return self.primary_model
    
    @override
    def invoke(
        self,
        input: LanguageModelInput,
        config: RunnableConfig | None = None,
        *,
        stop: list[str] | None = None,
        **kwargs,
    ) -> AIMessage:
        target_model = self._get_target_model()
        
        try:
            return target_model.invoke(input, config, stop=stop, **kwargs)
        except APIError as e:
            if getattr(e, "code", None) == 429 or getattr(e, "status", None) == 429 or str(429) in str(e):
                # Trigger 60-second cooldown on primary model
                self.rate_limit_reset_time = time.time() + 60.0
                return self.fallback_model.invoke(input, config, stop=stop, **kwargs)
            else:
                raise

    @override
    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable[..., Any] | BaseTool],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, AIMessage]:
        # Bind tools to BOTH models independently.
        # This prevents the wrapper from statically locking into one model
        # when bind_tools is called during app initialization.
        bound_primary = self.primary_model.bind_tools(tools, tool_choice=tool_choice, **kwargs)
        bound_fallback = self.fallback_model.bind_tools(tools, tool_choice=tool_choice, **kwargs)
        
        # Return a new instance of our wrapper wrapping the bound models
        wrapper = RateLimitedFallbackChatModel(
            primary_model=bound_primary,
            fallback_model=bound_fallback
        )
        wrapper.rate_limit_reset_time = self.rate_limit_reset_time
        return wrapper
    
    @override
    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        target_model = self._get_target_model()
        try:
            return target_model._generate(messages, stop, run_manager, **kwargs)
        except APIError as e:
            if getattr(e, "code", None) == 429 or getattr(e, "status", None) == 429 or str(429) in str(e):
                self.rate_limit_reset_time = time.time() + 60.0
                return self.fallback_model._generate(messages, stop, run_manager, **kwargs)
            else:
                raise
        
    def __getattr__(self, name):
        return getattr(self._get_target_model(), name)



def get_llm(provider: str = 'fallback'):
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
    
    if provider == 'fallback':
        return RateLimitedFallbackChatModel(
            primary_model=gemini,
            fallback_model=deepseek
        )
    elif provider == 'deepseek':
        return deepseek
    elif provider == 'google':
        return gemini
    else:
        raise ValueError(f"Unknown provider: {provider}")

llm = get_llm('fallback') # Use fallback model by default to handle Gemini rate limits
llm_with_tools = llm.bind_tools(tools)

system_prompt = """You are a helpful AI assistant connected to a local SQLite database sandbox.
You have access to tools to query the database.
Be brief in your responses."""

def call_model(state: AgentState):
    # Prepend the system prompt safely for this invocation
    messages = [SystemMessage(content=system_prompt)] + state.messages
    
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}
