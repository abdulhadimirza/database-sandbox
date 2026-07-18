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

def render_tool_call(console: Console, name: str, args: dict, result: str):
    try:
        args_str = json.dumps(args, indent=2)
    except Exception:
        args_str = str(args)
        
    console.print(Panel(
        f"[bold cyan]Tool Call:[/bold cyan] {name}\n[dim]{args_str}[/dim]\n\n[bold green]Result:[/bold green]\n[dim]{result}[/dim]",
        title="* Tool Execution",
        border_style="cyan"
    ))

@app.command()
def main():
    console.print(Panel.fit("[bold blue]Database Sandbox CLI[/bold blue]", border_style="blue"))
    
    session = ChatSession(agent)
    
    # Fetch chat history
    history = session.get_history()
    
    if history:
        console.print("[dim]Restoring previous session...[/dim]")
        for msg in history:
            if msg["role"] == "user":
                console.print(f"[bold green]You:[/bold green] {msg['content']}")
            elif msg["role"] == "assistant":
                content = msg.get("content")
                if content:
                    console.print("\n[bold blue]Assistant:[/bold blue]")
                    console.print(Markdown(content))
            elif msg["role"] == "tool":
                render_tool_call(console, msg['name'], msg['args'], msg['result'])
                
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
                    if event["type"] == "content":
                        full_response += event["content"]
                        live.update(Markdown(full_response + "|"))
                        
                    elif event["type"] == "tool":
                        if full_response:
                            live.update(Markdown(full_response))
                        else:
                            live.update("")
                        live.stop()
                        
                        render_tool_call(console, event['name'], event['args'], event['result'])
                        
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
