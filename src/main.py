import os
import sys

from dotenv import load_dotenv
load_dotenv()

def run_app() -> None:
    if "GOOGLE_API_KEY" not in os.environ:
        print("Warning: GOOGLE_API_KEY environment variable is not set.", file=sys.stderr)
        
    

if __name__ == "__main__":
    run_app()
