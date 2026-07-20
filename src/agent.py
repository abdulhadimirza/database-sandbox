import os
import sqlite3
from langgraph.graph import StateGraph, START
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.sqlite import SqliteSaver

from utils.state import AgentState
from utils.tools import tools
from utils.nodes import call_model

# Create the native LangGraph StateGraph
workflow = StateGraph(AgentState)

workflow.add_node('agent', call_model)
workflow.add_node('tools', ToolNode(tools))

workflow.add_edge(START, 'agent')
workflow.add_conditional_edges('agent', tools_condition)
workflow.add_edge('tools', 'agent')

# Implement SqliteSaver
DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'checkpoints.db')
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
memory = SqliteSaver(conn)

agent = workflow.compile(checkpointer=memory)
