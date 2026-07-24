import os
import sqlite3
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.sqlite import SqliteSaver

from utils.state import AgentState
from utils.tools import assistant_tools, editor_tools, generator_tools
from utils.nodes import create_agent_node
from utils.prompts import assistant_system_prompt, editor_system_prompt, generator_system_prompt

# Create agent nodes
database_assistant_node = create_agent_node(system_prompt=assistant_system_prompt, node_tools=assistant_tools)
data_editor_node = create_agent_node(system_prompt=editor_system_prompt, node_tools=editor_tools)
sample_generator_node = create_agent_node(system_prompt=generator_system_prompt, node_tools=generator_tools)

# Create the multi-agent LangGraph StateGraph
workflow = StateGraph(AgentState)

# Add Agent and Tool nodes
workflow.add_node('database_assistant_agent', database_assistant_node)
workflow.add_node('assistant_tools', ToolNode(assistant_tools))

workflow.add_node('data_editor_agent', data_editor_node)
workflow.add_node('editor_tools', ToolNode(editor_tools))

workflow.add_node('sample_generator_agent', sample_generator_node)
workflow.add_node('generator_tools', ToolNode(generator_tools))

# Add edges and conditional routing
workflow.add_edge(START, 'database_assistant_agent')

workflow.add_conditional_edges(
    'database_assistant_agent', 
    tools_condition,
    {'tools': 'assistant_tools', END: END}
)
workflow.add_edge('assistant_tools', 'database_assistant_agent')

workflow.add_conditional_edges(
    'data_editor_agent', 
    tools_condition,
    {'tools': 'editor_tools', END: END}
)
workflow.add_edge('editor_tools', 'data_editor_agent')

workflow.add_conditional_edges(
    'sample_generator_agent', 
    tools_condition,
    {'tools': 'generator_tools', END: END}
)
workflow.add_edge('generator_tools', 'sample_generator_agent')

# Implement SqliteSaver
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
raw_db_path = os.getenv('CHECKPOINT_DB_PATH', 'checkpoints.db')
DB_PATH = raw_db_path if os.path.isabs(raw_db_path) else os.path.abspath(os.path.join(PROJECT_ROOT, raw_db_path))
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
memory = SqliteSaver(conn)

agent = workflow.compile(checkpointer=memory)
