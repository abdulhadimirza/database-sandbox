from typing import Annotated, List, Optional
from pydantic import BaseModel
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

class AgentState(BaseModel):
    messages: Annotated[List[AnyMessage], add_messages]
    current_schema: Optional[str] = None
    active_query: Optional[str] = None
    errors: Optional[List[str]] = None
