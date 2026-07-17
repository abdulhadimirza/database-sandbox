import os
import sys
from streamlit.web import cli as stcli
from dotenv import load_dotenv

load_dotenv()

def run_app() -> None:
    if "GOOGLE_API_KEY" not in os.environ:
        print("Warning: GOOGLE_API_KEY environment variable is not set.", file=sys.stderr)
    
    # Programmatically run Streamlit pointing to src/app.py as the entry point
    sys.argv = ["streamlit", "run", "src/app.py"]
    sys.exit(stcli.main())

if __name__ == "__main__":
    run_app()

