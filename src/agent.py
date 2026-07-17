import os
import sqlite3
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent

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

# Initialize the model and tools
llm = ChatGoogleGenerativeAI(model=os.environ.get("MODEL"))
tools = [get_database_schema]

system_prompt = """You are a helpful AI assistant connected to a local SQLite database sandbox.
You have access to tools to query the database."""

# Create the LangGraph agent
agent = create_agent(llm, tools, system_prompt=system_prompt)
