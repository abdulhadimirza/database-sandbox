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
    """Safely execute one or more raw SQL queries provided by the LLM (separated by semicolons) and return the results."""
    statements = [stmt.strip() for stmt in query.split(";") if stmt.strip()]
    if not statements:
        return "No valid SQL statements found in input."

    try:
        with get_readonly_connection() as conn:
            conn.set_progress_handler(create_timeout_handler(2.0), 1000)
            cursor = conn.cursor()
            
            results_output = []
            for idx, stmt in enumerate(statements, 1):
                cursor.execute(stmt)
                rows = cursor.fetchmany(101)
                output_rows = rows[:100]
                
                stmt_hdr = f"Results for Statement {idx} ('{stmt}'):" if len(statements) > 1 else "Query Results:"
                if not output_rows:
                    results_output.append(f"{stmt_hdr}\nQuery executed successfully, but returned no rows.")
                else:
                    lines = [stmt_hdr]
                    for row in output_rows:
                        lines.append(str(dict(row)))
                    if len(rows) > 100:
                        lines.append("... Output truncated (100 rows maximum) ...")
                    results_output.append("\n".join(lines))
                    
            return "\n\n".join(results_output)
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
def analyze_query_impact(query: str) -> str:
    """Analyze one or more proposed write queries (separated by semicolons) by generating EXPLAIN QUERY PLAN and estimating affected row counts for each statement."""
    statements = [stmt.strip() for stmt in query.split(";") if stmt.strip()]
    if not statements:
        return "No valid SQL statements found in input."
        
    outputs = []
    try:
        with get_readonly_connection() as conn:
            cursor = conn.cursor()
            
            for idx, stmt in enumerate(statements, 1):
                stmt_output = [f"Statement {idx}: {stmt}"] if len(statements) > 1 else [f"Query: {stmt}"]
                
                has_error = False
                # EXPLAIN QUERY PLAN
                try:
                    cursor.execute(f"EXPLAIN QUERY PLAN {stmt}")
                    plan_rows = cursor.fetchall()
                    stmt_output.append("Query Plan:")
                    for row in plan_rows:
                        stmt_output.append(f"  detail: {row['detail']}")
                except Exception as plan_err:
                    has_error = True
                    stmt_output.append(f"Query Plan Error: {plan_err}")
                    
                # Estimate row count only if query plan succeeded
                if not has_error:
                    stmt_upper = stmt.upper()
                    row_count_info = ""
                    if stmt_upper.startswith("UPDATE") or stmt_upper.startswith("DELETE"):
                        try:
                            if stmt_upper.startswith("DELETE"):
                                from_idx = stmt_upper.find("FROM")
                                count_sql = "SELECT COUNT(*) as cnt " + stmt[from_idx:] if from_idx != -1 else ""
                            else:
                                parts = stmt.split("SET", 1)
                                table_part = parts[0].replace("UPDATE", "").strip()
                                where_part = ""
                                if "WHERE" in parts[1].upper():
                                    where_idx = parts[1].upper().find("WHERE")
                                    where_part = parts[1][where_idx:]
                                count_sql = f"SELECT COUNT(*) as cnt FROM {table_part} {where_part}"
                                
                            if count_sql:
                                cursor.execute(count_sql)
                                cnt = cursor.fetchone()["cnt"]
                                row_count_info = f"Estimated Affected Rows: {cnt}"
                        except Exception:
                            row_count_info = "Estimated Affected Rows: Unknown (could not parse row count pre-check)"
                    elif stmt_upper.startswith("INSERT"):
                        row_count_info = "Estimated Affected Rows: 1"
                    else:
                        row_count_info = "Estimated Affected Rows: N/A"
                        
                    if row_count_info:
                        stmt_output.append(row_count_info)
                else:
                    stmt_output.append("Estimated Affected Rows: N/A (Invalid Query)")
                    
                outputs.append("\n".join(stmt_output))

                
            return "\n\n".join(outputs)
    except Exception as e:
        raise ToolException(f"Error analyzing query impact: {e}")

@tool()
def execute_write_query(query: str, explanation: str) -> str:
    """Execute one or more raw SQL queries that modify the database (separated by semicolons). Requires a plain-English explanation of the blast radius / impact."""
    statements = [stmt.strip() for stmt in query.split(";") if stmt.strip()]
    if not statements:
        return "No valid SQL statements found in input."

    response = interrupt({
        "tool_name": "execute_write_query",
        "arguments": {"query": query, "explanation": explanation},
        "message": f"Approve executing the following SQL write query/queries?\n\nExplanation:\n{explanation}\n\nSQL:\n{query}"
    })
    
    if not isinstance(response, dict) or response.get("action") != "approve":
        return "Query execution cancelled by user."
        
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            total_rows_affected = 0
            for stmt in statements:
                cursor.execute(stmt)
                total_rows_affected += cursor.rowcount if cursor.rowcount > 0 else 0
            conn.commit()
            
            return f"All {len(statements)} statement(s) executed successfully. Total rows affected: {total_rows_affected}"
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
analyze_query_impact.handle_tool_error = True
execute_write_query.handle_tool_error = True
search_tables_by_keyword.handle_tool_error = True

tools = [
    list_tables, 
    describe_table, 
    execute_read_query,
    get_column_distinct_values,
    get_table_statistics,
    analyze_query_impact,
    execute_write_query,
    search_tables_by_keyword
]

