import os
import sqlite3
from typing import Annotated, List, Optional
from pydantic import BaseModel
from langchain_core.messages import AnyMessage, SystemMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

# The database file will be stored in the root directory
DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "sandbox.db")

@tool
def get_database_schema() -> str:
    """Returns the SQL schema for all tables in the local database. Use this to understand what tables and columns exist."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT name, sql FROM sqlite_master WHERE type='table';")
        schemas = []
        for name, sql in cursor.fetchall():
            if name and not name.startswith('sqlite_') and sql:
                schemas.append(sql)
        conn.close()
        
        if not schemas:
            return "No tables found in the database."
            
        return "\n\n".join(schemas)
    except Exception as e:
        return f"Error reading schema: {e}"

# Define the state schema
class AgentState(BaseModel):
    messages: Annotated[List[AnyMessage], add_messages]
    current_schema: Optional[str] = None
    active_query: Optional[str] = None
    errors: Optional[List[str]] = None

# Initialize the model and tools
llm = ChatGoogleGenerativeAI(model=os.environ.get("MODEL"))
tools = [get_database_schema]

system_prompt = """You are a helpful AI assistant connected to a local SQLite database sandbox.
You have access to tools to query the database."""

def call_model(state: AgentState):
    llm_with_tools = llm.bind_tools(tools)
    
    # Prepend the system prompt if not already present
    messages = state.messages
    if not any(isinstance(m, SystemMessage) for m in messages):
        messages = [SystemMessage(content=system_prompt)] + messages
        
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

# Create the native LangGraph StateGraph
workflow = StateGraph(AgentState)

workflow.add_node("agent", call_model)
workflow.add_node("tools", ToolNode(tools))

workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", tools_condition)
workflow.add_edge("tools", "agent")

# Implement MemorySaver
memory = MemorySaver()
agent = workflow.compile(checkpointer=memory)
