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

from chat_agent import (
    ChatAgent,
    ChatEvent,
    AgentThinkingEvent,
    AgentToolRequestEvent,
    AgentToolResultEvent,
    AgentToolErrorEvent,
    AgentMessageStartEvent,
    AgentMessageChunkEvent,
    AgentMessageCompleteEvent,
    UserMessageEvent,
    AgentTurnCompleteEvent,
)

console = Console()
app = typer.Typer()

def render_tool_request(console: Console, name: str, args: dict):
    try:
        args_str = json.dumps(args, indent=2)
    except Exception:
        args_str = str(args)
        
    console.print(Panel(
        f"[bold cyan]Tool Requested:[/bold cyan] {name}\n\n[bold green]Arguments:[/bold green]\n[dim]{args_str}[/dim]",
        title="* Tool Execution Requested",
        border_style="cyan"
    ))

def render_tool_result(console: Console, name: str, result: str):
    console.print(Panel(
        f"[bold cyan]Tool Responded:[/bold cyan] {name}\n\n[bold green]Result:[/bold green]\n[dim]{result}[/dim]",
        title="* Tool Execution Result",
        border_style="green"
    ))

class CLIRenderer:
    def __init__(self, console: Console):
        self.console = console
        self.live = None
        self.full_response = ""

    def start_live(self):
        if not self.live:
            self.live = Live(console=self.console, refresh_per_second=10)
            self.live.start()

    def stop_live(self):
        if self.live:
            if self.full_response:
                self.live.update(Markdown(self.full_response))
            else:
                self.live.update("")
            self.live.stop()
            self.live = None

    def handle_event(self, event: ChatEvent):
        if isinstance(event, AgentThinkingEvent):
            self.start_live()
            self.live.update(Spinner("dots", text="[dim]Thinking...[/dim]"))
            
        elif isinstance(event, AgentMessageStartEvent):
            pass
            
        elif isinstance(event, AgentMessageChunkEvent):
            self.full_response += event.chunk
            if self.live:
                self.live.update(Markdown(self.full_response + "|"))
                
        elif isinstance(event, AgentToolRequestEvent):
            self.stop_live()
            render_tool_request(self.console, event.tool_name, event.arguments)
            
        elif isinstance(event, AgentToolResultEvent):
            self.stop_live()
            render_tool_result(self.console, event.tool_name, event.result)
            # Reset response because the agent might say something after the tool
            self.full_response = ""
            
        elif isinstance(event, AgentToolErrorEvent):
            self.stop_live()
            render_tool_result(self.console, event.tool_name, f"Error: {event.error}")
            self.full_response = ""

        elif isinstance(event, AgentTurnCompleteEvent):
            self.stop_live()

@app.command()
def main():
    console.print(Panel.fit("[bold blue]Database Sandbox CLI[/bold blue]", border_style="blue"))
    
    agent = ChatAgent()
    renderer = CLIRenderer(console)
    agent.add_listener(renderer.handle_event)
    
    # Fetch chat history
    history = agent.get_history()
    
    if history:
        console.print("[dim]Restoring previous session...[/dim]")
        for event in history:
            if isinstance(event, UserMessageEvent):
                console.print(f"[bold green]You:[/bold green] {event.content}")
            elif isinstance(event, AgentMessageCompleteEvent):
                console.print("\n[bold blue]Assistant:[/bold blue]")
                console.print(Markdown(event.content))
            elif isinstance(event, AgentToolRequestEvent):
                render_tool_request(console, event.tool_name, event.arguments)
            elif isinstance(event, AgentToolResultEvent):
                render_tool_result(console, event.tool_name, event.result)
            elif isinstance(event, AgentToolErrorEvent):
                render_tool_result(console, event.tool_name, f"Error: {event.error}")
                
    while True:
        try:
            prompt = Prompt.ask("\n[bold green]You[/bold green]")
            if prompt.lower() in ['quit', 'exit', 'q']:
                break
            if not prompt.strip():
                continue
                
            console.print("\n[bold blue]Assistant:[/bold blue]")
            
            renderer.full_response = ""
            agent.send_message(prompt)
                
        except (KeyboardInterrupt, EOFError):
            break
            
    console.print("\n[bold blue]Goodbye![/bold blue]")

if __name__ == "__main__":
    app()
