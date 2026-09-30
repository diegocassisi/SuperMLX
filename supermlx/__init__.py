"""
SuperMLX — Agentic-optimized MLX inference server for Apple Silicon.

Keeps the KV cache alive across agent turns (radix prompt cache, tool-prefix cache,
cache canonicalization), with MTP speculative decoding, thinking control,
loop protection and Metal memory management.
"""

__version__ = "3.0.0"
