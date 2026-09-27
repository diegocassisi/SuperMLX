"""SuperMLX launcher — thin wrapper for backward compatibility.

Usage (from repo root):
    python SuperMLX.py                   # runs server.py (production)
    USE_SERVER2=true python SuperMLX.py  # runs server2.py (refactored)
    USE_SERVER3=true python SuperMLX.py  # runs server3.py (Fase D: RadixAttention + 4-phase)
"""
import os
from dotenv import load_dotenv
load_dotenv()

if os.environ.get("USE_SERVER3", "").lower() in ("1", "true", "yes"):
    from supermlx.server3 import run
elif os.environ.get("USE_SERVER2", "").lower() in ("1", "true", "yes"):
    from supermlx.server2 import run
else:
    from supermlx.server import run

if __name__ == "__main__":
    run()
