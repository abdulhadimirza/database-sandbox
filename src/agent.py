import os
import sqlite3
from langgraph.graph import StateGraph, START
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.sqlite import SqliteSaver

from utils.state import AgentState
from utils.tools import tools
from utils.nodes import create_agent_node
from utils.prompts import system_prompt

call_model = create_agent_node(system_prompt=system_prompt, node_tools=tools)

# Create the native LangGraph StateGraph
workflow = StateGraph(AgentState)

workflow.add_node('agent', call_model)
workflow.add_node('tools', ToolNode(tools))

workflow.add_edge(START, 'agent')
workflow.add_conditional_edges('agent', tools_condition)
workflow.add_edge('tools', 'agent')

# Implement SqliteSaver
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
raw_db_path = os.getenv('CHECKPOINT_DB_PATH', 'checkpoints.db')
DB_PATH = raw_db_path if os.path.isabs(raw_db_path) else os.path.abspath(os.path.join(PROJECT_ROOT, raw_db_path))
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
memory = SqliteSaver(conn)

agent = workflow.compile(checkpointer=memory)
