import time
from langchain_core.tools import tool, ToolException
from langgraph.types import interrupt

from database import get_readonly_connection, get_db_connection

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

@tool()
def get_column_distinct_values(table: str, column: str) -> str:
    """Queries and returns the distinct categorical values in a specified column."""
    try:
        with get_readonly_connection() as conn:
            cursor = conn.cursor()
            
            # Verify table exists to prevent SQL injection in pragma
            cursor.execute('SELECT name FROM sqlite_master WHERE type="table" AND name=?;', (table,))
            if not cursor.fetchone():
                return f"Table '{table}' does not exist."
                
            cursor.execute(f'PRAGMA table_info({table});')
            columns = [row['name'] for row in cursor.fetchall()]
            if column not in columns:
                return f"Column '{column}' does not exist in table '{table}'."
                
            query = f'SELECT DISTINCT {column} FROM {table} LIMIT 100;'
            cursor.execute(query)
            rows = cursor.fetchall()
            
            if not rows:
                return f"No values found in column '{column}'."
                
            values = [str(row[column]) for row in rows]
            return "\n".join(values)
    except Exception as e:
        raise ToolException(f"Error getting distinct values for '{table}.{column}': {e}")

@tool()
def get_table_statistics(table: str) -> str:
    """Returns row counts and basic bounds (min/max) for a specified table."""
    try:
        with get_readonly_connection() as conn:
            cursor = conn.cursor()
            
            # Verify table exists
            cursor.execute('SELECT name FROM sqlite_master WHERE type="table" AND name=?;', (table,))
            if not cursor.fetchone():
                return f"Table '{table}' does not exist."
                
            cursor.execute(f'PRAGMA table_info({table});')
            columns = cursor.fetchall()
            
            cursor.execute(f'SELECT COUNT(*) as count FROM {table};')
            count = cursor.fetchone()['count']
            
            output = [f"Statistics for table '{table}':", f"Total Rows: {count}"]
            
            numeric_types = ('INTEGER', 'REAL', 'NUMERIC')
            for col in columns:
                col_name = col['name']
                col_type = col['type'].upper()
                if any(t in col_type for t in numeric_types):
                    cursor.execute(f'SELECT MIN({col_name}) as min_val, MAX({col_name}) as max_val FROM {table};')
                    stats = cursor.fetchone()
                    output.append(f"  {col_name} ({col['type']}): MIN = {stats['min_val']}, MAX = {stats['max_val']}")
                    
            return "\n".join(output)
    except Exception as e:
        raise ToolException(f"Error getting statistics for '{table}': {e}")

@tool()
def execute_write_query(query: str) -> str:
    """Execute a raw SQL query that modifies the database (INSERT, UPDATE, DELETE, CREATE)."""
    response = interrupt({
        "tool_name": "execute_write_query",
        "arguments": {"query": query},
        "message": f"Approve executing the following SQL write query?\n\n{query}"
    })
    
    if not isinstance(response, dict) or response.get("action") != "approve":
        return "Query execution cancelled by user."
        
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query)
            conn.commit()
            
            return f"Query executed successfully. Rows affected: {cursor.rowcount}"
    except Exception as e:
        raise ToolException(f"Database Error: {e}")

@tool()
def search_tables_by_keyword(keyword: str) -> str:
    """Search for relevant tables based on a keyword match in table names or column names."""
    try:
        with get_readonly_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute('SELECT name FROM sqlite_master WHERE type="table";')
            tables = []
            for row in cursor.fetchall():
                name = row['name']
                if name and not name.startswith('sqlite_'):
                    tables.append(name)
                    
            matching_tables = set()
            keyword_lower = keyword.lower()
            
            for table in tables:
                if keyword_lower in table.lower():
                    matching_tables.add(table)
                    continue
                    
                cursor.execute(f'PRAGMA table_info({table});')
                columns = cursor.fetchall()
                for col in columns:
                    if keyword_lower in col['name'].lower():
                        matching_tables.add(table)
                        break
                        
            if not matching_tables:
                return f"No tables found matching keyword '{keyword}'."
                
            return "\n".join(sorted(list(matching_tables)))
    except Exception as e:
        raise ToolException(f"Error searching tables for keyword '{keyword}': {e}")

list_tables.handle_tool_error = True
describe_table.handle_tool_error = True
execute_read_query.handle_tool_error = True
get_column_distinct_values.handle_tool_error = True
get_table_statistics.handle_tool_error = True
execute_write_query.handle_tool_error = True
search_tables_by_keyword.handle_tool_error = True

tools = [
    list_tables, 
    describe_table, 
    execute_read_query,
    get_column_distinct_values,
    get_table_statistics,
    execute_write_query,
    search_tables_by_keyword
]
