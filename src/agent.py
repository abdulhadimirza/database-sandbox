import os
import time
from typing import Annotated, List, Optional
from pydantic import BaseModel
from langchain_core.messages import AnyMessage, SystemMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_deepseek import ChatDeepSeek
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from database import get_readonly_connection

@tool
def list_tables() -> str:
    """Query the database to return only a list of available table names."""
    try:
        with get_readonly_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT name FROM sqlite_master WHERE type="table";')
            tables = []
            for row in cursor.fetchall():
                name = row['name']
                if name and not name.startswith('sqlite_'):
                    tables.append(name)
            
            if not tables:
                return "No tables found in the database."
                
            return "\n".join(tables)
    except Exception as e:
        return f"Error reading tables: {e}"

@tool
def describe_table(table_name: str) -> str:
    """Given a table name, execute PRAGMA table_info and PRAGMA foreign_key_list to fetch the schema, AND run SELECT * FROM table_name LIMIT 3 to fetch a data sample."""
    try:
        with get_readonly_connection() as conn:
            cursor = conn.cursor()
            
            # Verify table exists to prevent SQL injection in pragma
            cursor.execute('SELECT name FROM sqlite_master WHERE type="table" AND name=?;', (table_name,))
            if not cursor.fetchone():
                return f"Table '{table_name}' does not exist."
                
            # Get schema
            cursor.execute(f'PRAGMA table_info({table_name});')
            columns = cursor.fetchall()
            
            # Get foreign keys
            cursor.execute(f'PRAGMA foreign_key_list({table_name});')
            fks = cursor.fetchall()
            
            # Get sample data
            cursor.execute(f'SELECT * FROM {table_name} LIMIT 3;')
            sample_rows = cursor.fetchall()
            
            output = [f"Schema for table '{table_name}':", "Columns:"]
            for col in columns:
                output.append(f"  {col['name']} ({col['type']})")
                
            if fks:
                output.append("Foreign Keys:")
                for fk in fks:
                    output.append(f"  {fk['from']} -> {fk['table']}({fk['to']})")
                    
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

def create_timeout_handler(timeout_seconds: float):
    start_time = time.time()
    def progress_handler() -> int:
        if time.time() - start_time > timeout_seconds:
            return 1 # Return non-zero to abort query
        return 0
    return progress_handler

@tool
def execute_read_query(query: str) -> str:
    """Safely execute a raw SQL query provided by the LLM and return the results."""
    try:
        with get_readonly_connection() as conn:
            conn.set_progress_handler(create_timeout_handler(2.0), 1000)
            cursor = conn.cursor()
            
            cursor.execute(query)
            # Fetch one extra row to detect if we hit the limit
            rows = cursor.fetchmany(101)
            
            output_rows = rows[:100]
            
            if not output_rows:
                return "Query executed successfully, but returned no rows."
                
            output = ["Query Results:"]
            for row in output_rows:
                output.append(str(dict(row)))
                
            if len(rows) > 100:
                output.append("... Output truncated (100 rows maximum) ...")
                
            return "\n".join(output)
    except Exception as e:
        return f"Database Error: {e}"

# Define the state schema
class AgentState(BaseModel):
    messages: Annotated[List[AnyMessage], add_messages]
    current_schema: Optional[str] = None
    active_query: Optional[str] = None
    errors: Optional[List[str]] = None

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
tools = [list_tables, describe_table, execute_read_query]

system_prompt = """You are a helpful AI assistant connected to a local SQLite database sandbox.
You have access to tools to query the database.
Be brief in your responses."""

llm_with_tools = llm.bind_tools(tools)

def call_model(state: AgentState):
    # Prepend the system prompt safely for this invocation
    messages = [SystemMessage(content=system_prompt)] + state.messages
    
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

# Create the native LangGraph StateGraph
workflow = StateGraph(AgentState)

workflow.add_node('agent', call_model)
workflow.add_node('tools', ToolNode(tools))

workflow.add_edge(START, 'agent')
workflow.add_conditional_edges('agent', tools_condition)
workflow.add_edge('tools', 'agent')

# Implement MemorySaver
memory = MemorySaver()
agent = workflow.compile(checkpointer=memory)
