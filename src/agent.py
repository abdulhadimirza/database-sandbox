import os
import sqlite3
import urllib.request
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
def list_tables() -> str:
    """Query the database to return only a list of available table names."""
    try:
        safe_path = urllib.request.pathname2url(DB_PATH)
        uri = f"file:{safe_path}?mode=ro"
        conn = sqlite3.connect(uri, uri=True)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = []
        for (name,) in cursor.fetchall():
            if name and not name.startswith('sqlite_'):
                tables.append(name)
        conn.close()
        
        if not tables:
            return "No tables found in the database."
            
        return "\n".join(tables)
    except Exception as e:
        return f"Error reading tables: {e}"

@tool
def describe_table(table_name: str) -> str:
    """Given a table name, execute PRAGMA table_info and PRAGMA foreign_key_list to fetch the schema, AND run SELECT * FROM table_name LIMIT 3 to fetch a data sample."""
    try:
        safe_path = urllib.request.pathname2url(DB_PATH)
        uri = f"file:{safe_path}?mode=ro"
        conn = sqlite3.connect(uri, uri=True)
        cursor = conn.cursor()
        
        # Verify table exists to prevent SQL injection in pragma
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?;", (table_name,))
        if not cursor.fetchone():
            return f"Table '{table_name}' does not exist."
            
        # Get schema
        cursor.execute(f"PRAGMA table_info({table_name});")
        columns = cursor.fetchall()
        
        # Get foreign keys
        cursor.execute(f"PRAGMA foreign_key_list({table_name});")
        fks = cursor.fetchall()
        
        # Get sample data
        cursor.execute(f"SELECT * FROM {table_name} LIMIT 3;")
        sample_rows = cursor.fetchall()
        
        conn.close()
        
        output = [f"Schema for table '{table_name}':", "Columns:"]
        for col in columns:
            output.append(f"  {col[1]} ({col[2]})")
            
        if fks:
            output.append("Foreign Keys:")
            for fk in fks:
                output.append(f"  {fk[3]} -> {fk[2]}({fk[4]})")
                
        output.append("Sample Data (max 3 rows):")
        for row in sample_rows:
            truncated_row = []
            for val in row:
                val_str = str(val)
                if len(val_str) > 50:
                    val_str = val_str[:47] + "..."
                truncated_row.append(val_str)
            output.append(f"  {truncated_row}")
            
        return "\n".join(output)
    except Exception as e:
        return f"Error describing table '{table_name}': {e}"

@tool
def execute_read_query(query: str) -> str:
    """Safely execute a raw SQL query provided by the LLM and return the results."""
    try:
        safe_path = urllib.request.pathname2url(DB_PATH)
        uri = f"file:{safe_path}?mode=ro"
        conn = sqlite3.connect(uri, uri=True)
        cursor = conn.cursor()
        
        cursor.execute(query)
        # Fetch one extra row to detect if we hit the limit
        rows = cursor.fetchmany(101)
        
        output_rows = rows[:100]
        conn.close()
        
        if not output_rows:
            return "Query executed successfully, but returned no rows."
            
        output = ["Query Results:"]
        for row in output_rows:
            output.append(str(row))
            
        if len(rows) > 100:
            output.append("... Output truncated (100 rows maximum) ...")
            
        return "\n".join(output)
    except sqlite3.Error as e:
        return f"Database Error: {e}"
    except Exception as e:
        return f"Unexpected Error: {e}"

# Define the state schema
class AgentState(BaseModel):
    messages: Annotated[List[AnyMessage], add_messages]
    current_schema: Optional[str] = None
    active_query: Optional[str] = None
    errors: Optional[List[str]] = None

# Initialize the model and tools
llm = ChatGoogleGenerativeAI(model=os.environ.get("MODEL"))
tools = [list_tables, describe_table, execute_read_query]

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
