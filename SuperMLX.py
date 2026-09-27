"""SuperMLX launcher — thin wrapper for backward compatibility.

Usage (from repo root):
    python SuperMLX.py                   # runs server.py (production)
    USE_SERVER2=true python SuperMLX.py  # runs server2.py (refactored)
"""
import os
from dotenv import load_dotenv
load_dotenv()

if os.environ.get("USE_SERVER2", "").lower() in ("1", "true", "yes"):
    from supermlx.server2 import run
else:
    from supermlx.server import run

if __name__ == "__main__":
    run()
