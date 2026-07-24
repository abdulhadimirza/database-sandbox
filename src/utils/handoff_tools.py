from typing import Optional, Annotated
from langchain_core.tools import tool, InjectedToolCallId
from langchain_core.messages import ToolMessage
from langgraph.types import Command

@tool()
def transfer_to_data_editor(
    reason: str,
    tool_call_id: Annotated[str, InjectedToolCallId] = ""
) -> Command:

    """
    Transfer the session to the Data Editor agent to perform database write/mutation operations (INSERT, UPDATE, DELETE, ALTER, etc.).
    
    Args:
        reason: Plain-English reason explaining why the write operation is being delegated.
    """
    return Command(
        goto="data_editor_agent",
        update={
            "current_agent": "data_editor_agent",
            "messages": [
                ToolMessage(
                    content=f"[System Navigation] Transferring control to Data Editor agent. Reason: {reason}",
                    tool_call_id=tool_call_id
                )
            ]
        }
    )

@tool()
def transfer_to_sample_generator(
    reason: str,
    target_table: Optional[str] = None,
    tool_call_id: Annotated[str, InjectedToolCallId] = ""
) -> Command:
    """
    Transfer the session to the Sample Data Generator agent to create synthetic mock data and populate database tables.
    
    Args:
        reason: Reason for initiating mock data generation.
        target_table: Optional target table name.
    """
    msg = f"[System Navigation] Transferring control to Sample Data Generator agent. Reason: {reason}"
    if target_table:
        msg += f" (Target Table: {target_table})"
        
    return Command(
        goto="sample_generator_agent",
        update={
            "current_agent": "sample_generator_agent",
            "messages": [
                ToolMessage(
                    content=msg,
                    tool_call_id=tool_call_id
                )
            ]
        }
    )

@tool()
def return_to_database_assistant(
    summary_of_work: str,
    tool_call_id: Annotated[str, InjectedToolCallId] = ""
) -> Command:
    """
    Return control back to the primary Database Assistant agent after completing a write or mock data task.
    
    Args:
        summary_of_work: Clear summary of the actions completed or cancelled.
    """
    return Command(
        goto="database_assistant_agent",
        update={
            "current_agent": "database_assistant_agent",
            "messages": [
                ToolMessage(
                    content=f"[System Navigation] Task complete. Returning control to Database Assistant. Summary: {summary_of_work}",
                    tool_call_id=tool_call_id
                )
            ]
        }
    )
