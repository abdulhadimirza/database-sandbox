import os
import json
import typer
from rich.console import Console
from rich.prompt import Prompt
from rich.markdown import Markdown
from rich.panel import Panel
from rich.live import Live
from rich.spinner import Spinner
from dotenv import load_dotenv

load_dotenv()

from agent import agent
from chat_session import ChatSession

console = Console()
app = typer.Typer()

@app.command()
def main():
    console.print(Panel.fit("[bold blue]Database Sandbox CLI[/bold blue]", border_style="blue"))
    
    session = ChatSession(agent)
    
    # Fetch chat history
    history = session.get_history()
    
    if history:
        console.print("[dim]Restoring previous session...[/dim]")
        pending_tool_calls = {}
        for msg in history:
            role = msg.get("role")
            if role == "user":
                console.print(f"[bold green]You:[/bold green] {msg.get('content')}")
            elif role == "assistant":
                content = msg.get("content")
                if content:
                    console.print("\n[bold blue]Assistant:[/bold blue]")
                    console.print(Markdown(content))
                tool_calls = msg.get("tool_calls", [])
                for tc in tool_calls:
                    pending_tool_calls[tc.get("id")] = tc
            elif role == "tool":
                tc_id = msg.get("id")
                tc = pending_tool_calls.get(tc_id, {})
                tc_args = tc.get("args", {})
                
                try:
                    args_str = json.dumps(tc_args, indent=2)
                except Exception:
                    args_str = str(tc_args)
                    
                console.print(Panel(
                    f"[bold cyan]Tool Call:[/bold cyan] {msg.get('name')}\n[dim]{args_str}[/dim]\n\n[bold green]Result:[/bold green]\n[dim]{msg.get('result')}[/dim]",
                    title="* Tool Execution",
                    border_style="cyan"
                ))
                
    while True:
        try:
            prompt = Prompt.ask("\n[bold green]You[/bold green]")
            if prompt.lower() in ['quit', 'exit', 'q']:
                break
            if not prompt.strip():
                continue
                
            console.print("\n[bold blue]Assistant:[/bold blue]")
            
            full_response = ""
            
            live = Live(console=console, refresh_per_second=10)
            live.update(Spinner("dots", text="[dim]Thinking...[/dim]"))
            live.start()
            
            try:
                for event in session.send_message(prompt):
                    if event["type"] == "content_chunk":
                        full_response += event["data"]
                        live.update(Markdown(full_response + "|"))
                        
                    elif event["type"] == "tool_execution":
                        if full_response:
                            live.update(Markdown(full_response))
                        else:
                            live.update("")
                        live.stop()
                        
                        console.print(Panel(
                            f"[bold cyan]Tool Call:[/bold cyan] {event.get('name')}\n[dim]{event.get('args')}[/dim]\n\n[bold green]Result:[/bold green]\n[dim]{event.get('result')}[/dim]",
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
                else:
                    live.update("")
                live.stop()
                
        except (KeyboardInterrupt, EOFError):
            break
            
    console.print("\n[bold blue]Goodbye![/bold blue]")

if __name__ == "__main__":
    app()
