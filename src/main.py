import os
import sys

from dotenv import load_dotenv
load_dotenv()

from agent import graph

def run_agentic_rag() -> None:
    if "GOOGLE_API_KEY" not in os.environ:
        print("Warning: GOOGLE_API_KEY environment variable is not set.", file=sys.stderr)
        
    print("Starting agentic RAG graph stream...\n")
    
    inputs = {
        "messages": [
            {
                "role": "user",
                "content": "What does Lilian Weng say about types of reward hacking?",
            }
        ]
    }
    
    # Stream the graph execution step by step
    for event in graph.stream(inputs):
        for node_name, output in event.items():
            print(f"--- Node: {node_name} ---")
            if "messages" in output and output["messages"]:
                last_message = output["messages"][-1]
                # Print the message cleanly
                last_message.pretty_print()
            print()

if __name__ == "__main__":
    run_agentic_rag()
