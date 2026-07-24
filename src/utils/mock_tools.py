import json
import random
from typing import Optional, Any
from faker import Faker
from langchain_core.tools import tool, ToolException

from database import get_readonly_connection, get_db_connection

fake = Faker()

@tool()
def generate_mock_records(table_name: str, num_records: int = 5, custom_rules: Optional[str] = None) -> str:
    """
    Generate synthetic mock data records for a specified database table based on its schema and foreign key constraints.
    Returns a JSON string containing the generated records ready for inspection or batch insertion.
    
    Args:
        table_name: Name of the target table.
        num_records: Number of mock records to generate (default: 5, max: 50).
        custom_rules: Optional JSON string of custom column rules (e.g. '{"status": ["active", "pending"]}').
    """
    try:
        num_records = max(1, min(num_records, 50))
        rules_dict = {}
        if custom_rules:
            try:
                rules_dict = json.loads(custom_rules) if isinstance(custom_rules, str) else custom_rules
            except Exception:
                pass

        with get_readonly_connection() as conn:
            cursor = conn.cursor()
            
            # Verify table exists
            cursor.execute('SELECT name FROM sqlite_master WHERE type="table" AND name=?;', (table_name,))
            if not cursor.fetchone():
                return f"Table '{table_name}' does not exist in the database."
                
            # Get table schema
            cursor.execute(f'PRAGMA table_info({table_name});')
            columns_info = cursor.fetchall()
            
            # Get foreign key constraints
            cursor.execute(f'PRAGMA foreign_key_list({table_name});')
            fks_info = cursor.fetchall()
            
            # Foreign key lookup map: col_name -> list of available parent values
            fk_map = {}
            for fk in fks_info:
                from_col = fk['from']
                parent_table = fk['table']
                to_col = fk['to']
                
                # Fetch available values from parent table
                cursor.execute(f'SELECT {to_col} FROM {parent_table} LIMIT 100;')
                parent_rows = cursor.fetchall()
                parent_vals = [row[to_col] for row in parent_rows if row[to_col] is not None]
                
                if not parent_vals:
                    return f"Cannot generate mock data for '{table_name}': Foreign key column '{from_col}' references '{parent_table}({to_col})', but '{parent_table}' has no records. Please generate mock data for '{parent_table}' first."
                
                fk_map[from_col] = parent_vals

            records = []
            for _ in range(num_records):
                record = {}
                for col in columns_info:
                    col_name = col['name']
                    col_type = col['type'].upper()
                    is_pk = col['pk'] > 0
                    not_null = col['notnull'] == 1
                    
                    # Skip autoincrement primary key columns
                    if is_pk and 'INT' in col_type:
                        continue
                        
                    # 1. Custom rule override
                    if col_name in rules_dict:
                        rule_val = rules_dict[col_name]
                        if isinstance(rule_val, list):
                            record[col_name] = random.choice(rule_val)
                        else:
                            record[col_name] = rule_val
                        continue
                        
                    # 2. Foreign Key constraint lookup
                    if col_name in fk_map:
                        record[col_name] = random.choice(fk_map[col_name])
                        continue

                    # 3. Semantic column name matching
                    name_lower = col_name.lower()
                    if name_lower == 'email':
                        record[col_name] = fake.email()
                    elif name_lower in ('name', 'full_name', 'user_name', 'username'):
                        record[col_name] = fake.name()
                    elif name_lower == 'first_name':
                        record[col_name] = fake.first_name()
                    elif name_lower == 'last_name':
                        record[col_name] = fake.last_name()
                    elif 'phone' in name_lower:
                        record[col_name] = fake.phone_number()
                    elif 'address' in name_lower:
                        record[col_name] = fake.address().replace('\n', ', ')
                    elif 'price' in name_lower or 'cost' in name_lower or 'amount' in name_lower or 'rate' in name_lower:
                        record[col_name] = round(random.uniform(5.0, 499.99), 2)
                    elif 'stock' in name_lower or 'quantity' in name_lower or 'count' in name_lower:
                        record[col_name] = random.randint(1, 100)
                    elif name_lower == 'status':
                        record[col_name] = random.choice(['pending', 'shipped', 'delivered', 'active'])
                    elif 'created_at' in name_lower or 'updated_at' in name_lower or 'timestamp' in name_lower or 'date' in name_lower:
                        record[col_name] = fake.date_time_this_year().strftime('%Y-%m-%d %H:%M:%S')
                    elif 'title' in name_lower:
                        record[col_name] = fake.sentence(nb_words=4).rstrip('.')
                    elif 'description' in name_lower:
                        record[col_name] = fake.paragraph(nb_sentences=2)

                    # 4. Data type fallback
                    else:
                        if 'INT' in col_type:
                            record[col_name] = random.randint(1, 100)
                        elif any(t in col_type for t in ('REAL', 'FLOAT', 'DOUBLE', 'NUMERIC')):
                            record[col_name] = round(random.uniform(1.0, 100.0), 2)
                        elif 'BOOL' in col_type:
                            record[col_name] = random.choice([0, 1])
                        else:
                            record[col_name] = fake.word().capitalize()

                records.append(record)

            return json.dumps({"table": table_name, "count": len(records), "records": records}, indent=2)

    except Exception as e:
        raise ToolException(f"Error generating mock records for '{table_name}': {e}")

@tool()
def batch_insert_mock_data(table_name: str, records_json: str) -> str:
    """
    Batch insert generated mock records into the specified database table within a single transaction.
    
    Args:
        table_name: Target table name.
        records_json: JSON string representing a list of record dicts (or output from generate_mock_records).
    """
    try:
        # Parse records
        if isinstance(records_json, str):
            data = json.loads(records_json)
            if isinstance(data, dict) and "records" in data:
                records = data["records"]
            elif isinstance(data, list):
                records = data
            else:
                return "Invalid records JSON structure. Expected a list of dictionaries or object with 'records'."
        elif isinstance(records_json, list):
            records = records_json
        else:
            return "Invalid input for records_json."

        if not records:
            return "No records provided for batch insertion."

        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Verify table exists
            cursor.execute('SELECT name FROM sqlite_master WHERE type="table" AND name=?;', (table_name,))
            if not cursor.fetchone():
                return f"Table '{table_name}' does not exist."
                
            cols = list(records[0].keys())
            cols_str = ", ".join(cols)
            placeholders = ", ".join(["?"] * len(cols))
            sql = f"INSERT INTO {table_name} ({cols_str}) VALUES ({placeholders});"

            rows_to_insert = [tuple(rec[col] for col in cols) for rec in records]
            
            cursor.executemany(sql, rows_to_insert)
            total_inserted = cursor.rowcount if cursor.rowcount > 0 else len(rows_to_insert)
            conn.commit()

            return f"Successfully batch inserted {total_inserted} mock record(s) into table '{table_name}'."

    except Exception as e:
        raise ToolException(f"Failed to batch insert mock data into '{table_name}': {e}")

generate_mock_records.handle_tool_error = True
batch_insert_mock_data.handle_tool_error = True
