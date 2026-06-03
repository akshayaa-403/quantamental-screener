# Deprecated - use app.py instead
import sys
from app import cli

if __name__ == "__main__":
    print("Note: main.py is deprecated. Use 'python app.py' instead.")
    sys.exit(cli())