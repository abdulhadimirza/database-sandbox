import time
from langchain_core.tools import tool, ToolException

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database import get_readonly_connection

@tool()
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
        raise ToolException(f"Error reading tables: {e}")

@tool()
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
        raise ToolException(f"Error describing table '{table_name}': {e}")

def create_timeout_handler(timeout_seconds: float):
    start_time = time.time()
    def progress_handler() -> int:
        if time.time() - start_time > timeout_seconds:
            return 1 # Return non-zero to abort query
        return 0
    return progress_handler

@tool()
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
        raise ToolException(f"Database Error: {e}")

list_tables.handle_tool_error = True
describe_table.handle_tool_error = True
execute_read_query.handle_tool_error = True

tools = [list_tables, describe_table, execute_read_query]
