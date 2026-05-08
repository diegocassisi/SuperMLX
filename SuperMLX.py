"""SuperMLX launcher — thin wrapper for backward compatibility.

Usage (from repo root):
    python SuperMLX.py

Equivalent to:
    python -m supermlx.server
"""
from supermlx.server import run

if __name__ == "__main__":
    run()
