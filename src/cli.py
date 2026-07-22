from dotenv import load_dotenv
load_dotenv()

import json
import shutil
import typer
from rich.console import Console
from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.styles import Style
from rich.markdown import Markdown
from rich.panel import Panel
from rich.live import Live
from rich.spinner import Spinner

from chat_agent import (
    ChatAgent,
    ChatEvent,
    AgentThinkingEvent,
    AgentToolRequestEvent,
    AgentToolResultEvent,
    AgentToolErrorEvent,
    AgentToolApprovalRequestEvent,
    AgentMessageStartEvent,
    AgentMessageChunkEvent,
    AgentMessageCompleteEvent,
    UserMessageEvent,
    AgentTurnCompleteEvent,
    AgentErrorEvent,
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
        border_style='cyan'
    ))

def render_tool_approval_request(console: Console, name: str, args: dict, message: str):
    try:
        args_str = json.dumps(args, indent=2)
    except Exception:
        args_str = str(args)
        
    console.print(Panel(
        f"[bold yellow]Approval Required for Tool:[/bold yellow] {name}\n\n[bold white]{message}[/bold white]\n\n[bold green]Arguments:[/bold green]\n[dim]{args_str}[/dim]\n\n[bold yellow]Type 'y' / 'yes' to approve, or anything else to cancel.[/bold yellow]",
        title="! Action Paused - Human Approval Required",
        border_style='yellow'
    ))

def render_tool_result(console: Console, name: str, result: str):
    console.print(Panel(
        f"[bold cyan]Tool Responded:[/bold cyan] {name}\n\n[bold green]Result:[/bold green]\n[dim]{result}[/dim]",
        title="* Tool Execution Result",
        border_style='green'
    ))

def render_tool_error(console: Console, name: str, error: str):
    console.print(Panel(
        f"[bold cyan]Tool Failed:[/bold cyan] {name}\n\n[bold red]Error:[/bold red]\n[dim]{error}[/dim]",
        title="* Tool Execution Error",
        border_style='red'
    ))

def render_agent_error(console: Console, error: str):
    console.print(Panel(
        f"[bold red]Agent Error:[/bold red]\n[dim]{error}[/dim]",
        title="* Error",
        border_style='red'
    ))

class CLIRenderer:
    def __init__(self, console: Console):
        self.console = console
        self.live = None
        self.full_response = ''

    def _debug(self, msg):
        pass#print(msg)

    def start_live(self):
        self._debug("start_live called")
        if not self.live:
            self.live = Live(
                console=self.console, 
                refresh_per_second=10, 
                transient=True, 
                vertical_overflow="visible"
            )
            self.live.start()

    def stop_live(self):
        self._debug(f"stop_live called. live active: {bool(self.live)}")
        if self.live:
            self.live.stop()
            self.live = None

    def handle_event(self, event: ChatEvent):
        self._debug(f"handle_event: {type(event).__name__}")
        if isinstance(event, UserMessageEvent):
            self.stop_live()
            self.console.print(f"\n[bold green]You:[/bold green]\n{event.content}")
            self.console.print("\n[bold blue]Assistant:[/bold blue]")

        elif isinstance(event, AgentThinkingEvent):
            self.start_live()
            self.live.update(Spinner('dots', text="[dim]Thinking...[/dim]"))
            
        elif isinstance(event, AgentMessageStartEvent):
            self.full_response = ''
            
        elif isinstance(event, AgentMessageChunkEvent):
            self.full_response += event.chunk
            if not self.live:
                self.start_live()
                
            term_height = shutil.get_terminal_size().lines
            max_lines = max(5, term_height - 10)
            
            lines = self.full_response.split("\n")
            if len(lines) > max_lines:
                display_text = "...\n" + "\n".join(lines[-max_lines:])
            else:
                display_text = self.full_response
                
            self.live.update(Markdown(display_text + " ▌"))
            
        elif isinstance(event, AgentMessageCompleteEvent):
            self.stop_live()
            display_text = self.full_response if self.full_response else getattr(event, 'content', '')
            if display_text:
                self.console.print(Markdown(display_text))
            self.full_response = ''
            
        elif isinstance(event, AgentToolRequestEvent):
            self.stop_live()
            if self.full_response:
                self.console.print(Markdown(self.full_response))
            render_tool_request(self.console, event.tool_name, event.arguments)
            self.full_response = ''
            
        elif isinstance(event, AgentToolApprovalRequestEvent):
            self.stop_live()
            if self.full_response:
                self.console.print(Markdown(self.full_response))
            render_tool_approval_request(self.console, event.tool_name, event.arguments, event.message)
            self.full_response = ''

        elif isinstance(event, AgentToolResultEvent):
            self.stop_live()
            render_tool_result(self.console, event.tool_name, event.result)
            self.full_response = ''
            
        elif isinstance(event, AgentToolErrorEvent):
            self.stop_live()
            render_tool_error(self.console, event.tool_name, event.error)
            self.full_response = ''

        elif isinstance(event, AgentErrorEvent):
            self.stop_live()
            render_agent_error(self.console, event.error)
            self.full_response = ''

        elif isinstance(event, AgentTurnCompleteEvent):
            self.stop_live()
            if self.full_response:
                self.console.print(Markdown(self.full_response))
            self.full_response = ''

@app.command()
def main():
    console.print(Panel.fit("[bold blue]Database Sandbox CLI[/bold blue]", border_style='blue'))
    
    agent = ChatAgent()
    renderer = CLIRenderer(console)
    agent.add_listener(renderer.handle_event)
    
    # Restore chat history organically via listener events
    console.print("[dim]Restoring previous session...[/dim]")
    agent.load()
                
    session = PromptSession()
    
    bindings = KeyBindings()

    @bindings.add('enter')
    def _(event):
        event.current_buffer.validate_and_handle()

    @bindings.add('escape', 'enter')
    def _(event):
        event.current_buffer.insert_text('\n')
        
    while True:
        try:
            if agent.is_paused:
                prompt_label = HTML("\n<ansiyellow><b>Approve execution? (y/n):</b></ansiyellow>\n")
                toolbar_label = HTML("<b>Type 'y' or 'yes' to approve | Any other input to cancel/reject | /quit or /exit to exit</b>")
            else:
                prompt_label = HTML("\n<ansigreen><b>You:</b></ansigreen>\n")
                toolbar_label = HTML("<b>[Enter] to send | [Esc] -> [Enter] for new line | /quit or /exit to exit</b>")
                
            prompt = session.prompt(
                prompt_label,
                multiline=True,
                key_bindings=bindings,
                bottom_toolbar=toolbar_label,
                style=Style.from_dict({'bottom-toolbar': 'default'})
            )
            
            if prompt.strip() in ['/quit', '/exit']:
                break
            if not prompt.strip():
                continue
                
            console.print("\n[bold blue]Assistant:[/bold blue]")
            
            renderer.full_response = ''
            if agent.is_paused:
                approved = prompt.strip().lower() in ['y', 'yes']
                agent.respond_to_approval(approved)
            else:
                agent.send_message(prompt)
                
        except (KeyboardInterrupt, EOFError):
            break
            
    console.print("\n[bold blue]Goodbye![/bold blue]")

if __name__ == '__main__':
    app()
