import ingest
import search_agent 
import logs

import asyncio

from logger import logger

REPO_OWNER = "pydantic"
REPO_NAME = "pydantic-ai"
CHUNK_METHOD = "markdown_sections"


def initialize_index():
    logger.info(f"Starting AI Pydantic-AI Assistant for {REPO_OWNER}/{REPO_NAME}")
    logger.info("Initializing data ingestion...")

    index = ingest.index_data(REPO_OWNER, REPO_NAME, chunk_method=CHUNK_METHOD)
    logger.info("Data indexing completed successfully!")
    return index


def initialize_agent(index):
    logger.info("Initializing search agent...")
    agent = search_agent.init_agent(index, REPO_OWNER, REPO_NAME)
    logger.info("Agent initialized successfully!")
    return agent

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.markdown import Markdown
from rich.text import Text

console = Console()

def main():
    index = initialize_index()
    agent = initialize_agent(index)
    
    console.print(Panel("🤖 [bold cyan]Pydantic AI Assistant[/bold cyan]\nAsk me anything about the repo!", style="bold green"))
    console.print("Type '[bold red]stop[/bold red]' to exit.\n")

    while True:
        # Get user input with styled prompt
        question = Prompt.ask("[bold yellow]Your question[/bold yellow]")
        
        if question.strip().lower() == 'stop':
            console.print("\n[bold red]Goodbye! 👋[/bold red]")
            break

        console.print("\n[blue]Processing your question...[/blue]\n")

        try:
            response = asyncio.run(agent.run(user_prompt=question))
            logs.log_interaction_to_file(agent, response.new_messages())

            # Display response nicely with a panel
            console.print(Panel.fit(
                Markdown(response.output),
                title="🤖 Pydantic AI",
                border_style="cyan",
                padding=(1, 2)
            ))
            console.print(Text("="*50, style="dim"))

        except Exception as e:
            console.print(Panel(f"[red]⚠️ Error:[/red] {e}", style="bold red"))


if __name__ == "__main__":
    main()