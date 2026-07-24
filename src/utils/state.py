from typing import Annotated, List, Optional
from pydantic import BaseModel
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

class AgentState(BaseModel):
    messages: Annotated[List[AnyMessage], add_messages]
    current_agent: Optional[str] = None

