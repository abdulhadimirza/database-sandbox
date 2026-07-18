import os
import uuid
import json
import typer
from rich.console import Console
from rich.prompt import Prompt
from rich.markdown import Markdown
from rich.panel import Panel
from rich.live import Live
from rich.spinner import Spinner
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, AIMessageChunk

load_dotenv()

from agent import agent

console = Console()
app = typer.Typer()

@app.command()
def main():
    console.print(Panel.fit("[bold blue]Database Sandbox CLI[/bold blue]", border_style="blue"))
    
    # Initialize thread_id for LangGraph MemorySaver
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    
    # Fetch chat history from LangGraph checkpointer
    state = agent.get_state(config)
    messages = state.values.get("messages", []) if state and hasattr(state, 'values') and state.values else []
    
    if messages:
        console.print("[dim]Restoring previous session...[/dim]")
        for msg in messages:
            if isinstance(msg, HumanMessage):
                console.print(f"[bold green]You:[/bold green] {msg.content}")
            elif isinstance(msg, AIMessage):
                if msg.content:
                    if isinstance(msg.content, str):
                        console.print(Markdown(msg.content))
                    elif isinstance(msg.content, list):
                        for block in msg.content:
                            if isinstance(block, dict) and block.get("type") == "text" and "text" in block:
                                console.print(Markdown(block["text"]))
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    for tc in msg.tool_calls:
                        console.print(f"[dim cyan]> Tool Call: {tc['name']}[/dim cyan]")
            elif isinstance(msg, ToolMessage):
                console.print(f"[dim green]+ Tool Result: {msg.name}[/dim green]")
                
    while True:
        try:
            prompt = Prompt.ask("\n[bold green]You[/bold green]")
            if prompt.lower() in ['quit', 'exit', 'q']:
                break
            if not prompt.strip():
                continue
                
            input_state = {"messages": [HumanMessage(content=prompt)]}
            
            console.print("\n[bold blue]Assistant:[/bold blue]")
            
            full_response = ""
            active_tool_calls = {}
            
            live = Live(console=console, refresh_per_second=10)
            live.update(Spinner("dots", text="[dim]Thinking...[/dim]"))
            live.start()
            
            try:
                for chunk in agent.stream(input_state, config, stream_mode="messages", version="v2"):
                    if chunk["type"] == "messages":
                        msg, metadata = chunk["data"]
                        
                        if isinstance(msg, (AIMessage, AIMessageChunk)):
                            if msg.content:
                                if isinstance(msg.content, str):
                                    full_response += msg.content
                                elif isinstance(msg.content, list):
                                    for block in msg.content:
                                        if isinstance(block, dict) and block.get("type") == "text" and "text" in block:
                                            full_response += block["text"]
                                live.update(Markdown(full_response + "|"))
                                
                            if hasattr(msg, "tool_call_chunks") and msg.tool_call_chunks:
                                for tc in msg.tool_call_chunks:
                                    idx = tc.get("index")
                                    if idx not in active_tool_calls:
                                        active_tool_calls[idx] = {"name": tc.get("name", ""), "args": "", "id": tc.get("id", "")}
                                    if tc.get("args"):
                                        active_tool_calls[idx]["args"] += tc["args"]
                                
                        elif isinstance(msg, ToolMessage):
                            if full_response:
                                live.update(Markdown(full_response))
                            live.stop()
                            
                            # Find matching tool call to display arguments
                            tc_args = "{}"
                            for tc in active_tool_calls.values():
                                if tc["id"] == getattr(msg, "tool_call_id", "") or tc["name"] == msg.name:
                                    tc_args = tc["args"]
                                    break
                            
                            try:
                                parsed_args = json.loads(tc_args) if tc_args else {}
                                args_str = json.dumps(parsed_args, indent=2)
                            except json.JSONDecodeError:
                                args_str = tc_args
                                
                            console.print(Panel(
                                f"[bold cyan]Tool Call:[/bold cyan] {msg.name}\n[dim]{args_str}[/dim]\n\n[bold green]Result:[/bold green]\n[dim]{msg.content}[/dim]",
                                title="* Tool Execution",
                                border_style="cyan"
                            ))
                            
                            full_response = ""
                            live = Live(console=console, refresh_per_second=10)
                            live.update(Spinner("dots", text="[dim]Thinking...[/dim]"))
                            live.start()
                            
            finally:
                if full_response:
                    live.update(Markdown(full_response))
                live.stop()
                
        except (KeyboardInterrupt, EOFError):
            break
            
    console.print("\n[bold blue]Goodbye![/bold blue]")

if __name__ == "__main__":
    app()
