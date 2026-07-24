import os
import sqlite3
from typing import Optional
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.sqlite import SqliteSaver

from utils.tools import assistant_tools as base_assistant_tools, editor_tools, generator_tools
from utils.nodes import create_agent_node
from utils.prompts import assistant_system_prompt, editor_system_prompt, generator_system_prompt

# 1. Create Data Editor Subgraph
editor_node = create_agent_node(system_prompt=editor_system_prompt, node_tools=editor_tools)
editor_workflow = StateGraph(MessagesState)
editor_workflow.add_node('agent', editor_node)
editor_workflow.add_node('tools', ToolNode(editor_tools))
editor_workflow.add_edge(START, 'agent')
editor_workflow.add_conditional_edges('agent', tools_condition, {'tools': 'tools', END: END})
editor_workflow.add_edge('tools', 'agent')
data_editor_graph = editor_workflow.compile()

# 2. Create Sample Data Generator Subgraph
generator_node = create_agent_node(system_prompt=generator_system_prompt, node_tools=generator_tools)
generator_workflow = StateGraph(MessagesState)
generator_workflow.add_node('agent', generator_node)
generator_workflow.add_node('tools', ToolNode(generator_tools))
generator_workflow.add_edge(START, 'agent')
generator_workflow.add_conditional_edges('agent', tools_condition, {'tools': 'tools', END: END})
generator_workflow.add_edge('tools', 'agent')
sample_generator_graph = generator_workflow.compile()

# 3. Define Subagent Tools for Assistant
@tool
def call_data_editor(query: str, config: Optional[RunnableConfig] = None) -> str:
    """
    Delegate database write or mutation operations (INSERT, UPDATE, DELETE, ALTER, DROP, etc.) to the Data Editor subagent.
    
    Args:
        query: Clear instructions or SQL statement for the write operation.
    """
    result = data_editor_graph.invoke({"messages": [("user", query)]}, config)
    messages = result.get("messages", [])
    if messages:
        return messages[-1].content
    return "Data Editor task completed."

@tool
def call_sample_generator(target_table: str, requirements: Optional[str] = None, config: Optional[RunnableConfig] = None) -> str:
    """
    Delegate synthetic mock data generation and populating database tables to the Sample Data Generator subagent.
    
    Args:
        target_table: Name of the table to generate sample data for.
        requirements: Optional custom rules or specific column guidelines.
    """
    prompt = f"Generate mock data for table '{target_table}'."
    if requirements:
        prompt += f" Requirements/Rules: {requirements}"
    result = sample_generator_graph.invoke({"messages": [("user", prompt)]}, config)
    messages = result.get("messages", [])
    if messages:
        return messages[-1].content
    return "Sample Data Generator task completed."

# Combine base read-only tools with subagent tools
assistant_tools = base_assistant_tools + [call_data_editor, call_sample_generator]

# 4. Create Main Database Assistant Graph
assistant_node = create_agent_node(system_prompt=assistant_system_prompt, node_tools=assistant_tools)
main_workflow = StateGraph(MessagesState)
main_workflow.add_node('database_assistant_agent', assistant_node)
main_workflow.add_node('assistant_tools', ToolNode(assistant_tools))
main_workflow.add_edge(START, 'database_assistant_agent')
main_workflow.add_conditional_edges(
    'database_assistant_agent', 
    tools_condition,
    {'tools': 'assistant_tools', END: END}
)
main_workflow.add_edge('assistant_tools', 'database_assistant_agent')

# Implement SqliteSaver Checkpointer
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
raw_db_path = os.getenv('CHECKPOINT_DB_PATH', 'checkpoints.db')
DB_PATH = raw_db_path if os.path.isabs(raw_db_path) else os.path.abspath(os.path.join(PROJECT_ROOT, raw_db_path))
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
memory = SqliteSaver(conn)

agent = main_workflow.compile(checkpointer=memory)

