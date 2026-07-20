from langgraph.graph import StateGraph, START
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

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

# Implement MemorySaver
memory = MemorySaver()
agent = workflow.compile(checkpointer=memory)
