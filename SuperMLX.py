"""SuperMLX launcher — thin wrapper.

Usage (from repo root):
    python SuperMLX.py

Legacy servers (old server.py / server2.py) live in _attic/supermlx/.
"""
from dotenv import load_dotenv
load_dotenv()

from supermlx.server import run

if __name__ == "__main__":
    run()
