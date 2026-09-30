"""
rag_enricher.py — RAG Context Enricher for SuperMLX Server
=================================================================
Enriches prompts with relevant codebase context before sending
them to the model, eliminating the need for the model to "guess"
APIs and structures.

Architecture:
    CPU:   Qwen3-Embedding-0.6B (embeddings, ~1.2GB, no Metal)
    Disk:  LanceDB embedded (vector search, no server)
    Metal: 100% free for model inference

Inspired by OpenClaw RAG Proxy but inverted: enriches instead of
compressing.
"""

import os
import re
import json
import time
import hashlib
import logging
import pathlib
from typing import List, Dict, Optional, Tuple

from .config import SETTINGS

logger = logging.getLogger("rag_enricher")

# ── Configuration ─────────────────────────────────────────────────────────────

# Embedding model — corre en CPU, no compite con Metal
# Qwen3-Embedding-0.6B: MMTEB multilingual 70.7 (top <1B params, 2025)
# Superior cross-lingual ES→EN vs nomic/BGE-M3 (+7 pts MMTEB).
# ~1.2GB fp16 | 60-120ms CPU (M4 Pro) | latencia irrelevante vs TTFT de Qwen (2-10s)
# sentence-transformers >=2.7.0 requerido (instalado: 5.2.0 ✅)
EMBED_MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"

# LanceDB storage — embebido, sin servidor
LANCEDB_PATH = str(pathlib.Path(__file__).parent / "rag_data")
TABLE_NAME = "codebase_chunks"

# ── Switches (hardcoded, change here) ─────────────────────────────────────────
ENRICHER_ENABLED = True      # Inject codebase context into technical prompts
COMPRESSOR_ENABLED = True    # Compress long conversation history

# Enricher config
TOP_K = 5                   # 5 chunks × ~2400 tok = ~12000 tok max injectable
CHUNK_SIZE_LINES = 80       # 80 lines: semantic precision confirmed in benchmark
MAX_CONTEXT_TOKENS = 65535  # Safety cap. With TOP_K=5 ~12000 tok arrive → cap is not the real limit.
                            # Raised from TOP_K=2 → 5: model requested more context to avoid tool calls
RELEVANCE_THRESHOLD = 1.6   # Max L2 distance. 1.3→1.6: Qwen3-Embedding asymmetric
                            # (queries with Instruct: prefix, documents without) produces
                            # slightly higher distances than nomic. Max L2=1.414.

# Compressor config
COMPRESSION_THRESHOLD = 3000 # Tokens a partir de los cuales comprimir historial
COMPRESSOR_TOP_K = 10        # Retrieve more so the reranker can filter (was 3)
COMPRESSOR_RERANKER_TOP_N = 4  # Chunks to keep after reranking
COMPRESSOR_GUARD = 4         # Last N messages that are NEVER compressed
CHAT_TABLE_NAME = "chat_sessions"
CHUNK_SIZE_TOKENS = 600      # Chunk size for history

# LLMLingua-2 — fine-grained token compression (CPU, does not touch Metal)
COMPRESSOR_LLMLINGUA_ENABLED = True
COMPRESSOR_LLMLINGUA_MODEL = "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank"
COMPRESSOR_LLMLINGUA_RATIO = 0.5  # Comprimir al 50%

# Reranker — poda gruesa de chunks por relevancia (CPU)
COMPRESSOR_RERANKER_ENABLED = True
COMPRESSOR_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # 22M params, ~80MB

# Code extensions to index
CODE_EXTENSIONS = {
    # Code
    ".py", ".json", ".yaml", ".yml", ".toml",
    # Documentation
    ".md", ".txt", ".rtf",
    # Hardware / embebidos
    ".ino", ".h", ".cpp", ".c",
}

# Re-index on every startup (drops and recreates the index). Env: RAG_FORCE_REINDEX=true
FORCE_REINDEX = SETTINGS.rag_force_reindex

# Archivos/dirs a excluir del index
EXCLUDE_PATTERNS = {
    "__pycache__", ".git", ".venv", "venv", "node_modules",
    ".DS_Store", "backup", "Legacy", ".venv_mlx_bonsai",
}

# ── Patrones de ruido OpenClaw (mismo set que openclaw_rag_proxyV2 V3.6) ─────
_RE_SECURITY_BLOCK = re.compile(
    r'<<<SECURITY NOTICE.*?END_EXTERNAL_UNTRUSTED_CONTENT[^>]*>>>',
    re.DOTALL
)
_RE_OPENCLAW_SENDER = re.compile(
    r'Sender \(untrusted metadata\):\s*```json\s*\{[^}]*\}\s*```\s*',
    re.DOTALL
)
_RE_OPENCLAW_TIMESTAMP = re.compile(
    r'\[.*?\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}\s+GMT[+-]\d+\]\s*'
)
_SKIP_ROLES = {"tool"}


def _sanitize_content_for_rag(content: str) -> str:
    """Strip OpenClaw noise before indexing in LanceDB."""
    if not content or not isinstance(content, str):
        return content or ""
    content = _RE_SECURITY_BLOCK.sub("", content)
    content = _RE_OPENCLAW_SENDER.sub("", content)
    content = _RE_OPENCLAW_TIMESTAMP.sub("", content)
    content = re.sub(r'\n{3,}', '\n\n', content).strip()
    return content


# ── Global State (lazy init) ─────────────────────────────────────────────────

_embedder = None
_lance_db = None
_lance_table = None
_indexed = False
_reranker = None       # CrossEncoder para reranking
_llmlingua = None      # LLMLingua-2 for fine-grained compression


def _get_embedder():
    """Load embedding model on CPU. Lazy init, once."""
    global _embedder
    if _embedder is not None:
        return _embedder

    try:
        from sentence_transformers import SentenceTransformer
        import torch

        logger.info("🧠 [RAG] Loading embedding model on CPU: %s", EMBED_MODEL_NAME)
        t0 = time.time()

        _embedder = SentenceTransformer(
            EMBED_MODEL_NAME,
            device="cpu",           # CPU — Metal 100% libre para Qwen + TurboQuant (delta ~30-60ms, irrelevante vs TTFT)
            trust_remote_code=True,
        )

        elapsed = time.time() - t0
        logger.info("✅ [RAG] Embedding model ready (%.1fs, CPU)", elapsed)
        return _embedder

    except ImportError:
        logger.warning("⚠️ [RAG] sentence-transformers not installed. RAG disabled.")
        return None
    except Exception as e:
        logger.error("❌ [RAG] Error loading embeddings: %s", e)
        return None


def _get_lance():
    """Lazy init de LanceDB."""
    global _lance_db, _lance_table
    if _lance_db is not None:
        return _lance_db, _lance_table

    try:
        import lancedb

        os.makedirs(LANCEDB_PATH, exist_ok=True)
        _lance_db = lancedb.connect(LANCEDB_PATH)

        if TABLE_NAME in _lance_db.table_names():
            _lance_table = _lance_db.open_table(TABLE_NAME)
            logger.info("[RAG] LanceDB tabla '%s' cargada (%d chunks)",
                        TABLE_NAME, _lance_table.count_rows())
        else:
            _lance_table = None
            logger.info("[RAG] LanceDB table '%s' does not exist yet. Will create on indexing.", TABLE_NAME)

        return _lance_db, _lance_table

    except ImportError:
        logger.warning("⚠️ [RAG] lancedb no instalado. RAG deshabilitado.")
        return None, None
    except Exception as e:
        logger.error("❌ [RAG] Error conectando LanceDB: %s", e)
        return None, None


# ── Codebase Indexing ─────────────────────────────────────────────────────────

def _should_exclude(path: pathlib.Path) -> bool:
    """Check if a file/directory should be excluded from the index."""
    for part in path.parts:
        if part in EXCLUDE_PATTERNS:
            return True
    return False


def _chunk_file(filepath: pathlib.Path, chunk_size: int = CHUNK_SIZE_LINES) -> List[Dict]:
    """Split a file into coherent chunks with metadata."""
    try:
        content = filepath.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return []

    lines = content.splitlines()
    if not lines:
        return []

    # Full file hash — to detect changes in incremental indexing
    file_hash = hashlib.sha256(content.encode()).hexdigest()[:16]

    chunks = []
    for i in range(0, len(lines), chunk_size):
        chunk_lines = lines[i:i + chunk_size]
        chunk_text = "\n".join(chunk_lines)

        if len(chunk_text.strip()) < 20:    # Skip empty chunks
            continue

        chunks.append({
            "text": chunk_text,
            "filepath": str(filepath),
            "filename": filepath.name,
            "file_hash": file_hash,
            "start_line": i + 1,
            "end_line": min(i + chunk_size, len(lines)),
            "total_lines": len(lines),
            "extension": filepath.suffix,
            "chunk_id": hashlib.sha256(chunk_text.encode()).hexdigest()[:12],
        })

    return chunks


def _collect_files(root_dir: str, extensions: set = CODE_EXTENSIONS) -> List[pathlib.Path]:
    """Collect indexable files from the codebase."""
    root = pathlib.Path(root_dir)
    files = []

    for filepath in root.rglob("*"):
        if not filepath.is_file():
            continue
        if filepath.suffix not in extensions:
            continue
        if _should_exclude(filepath):
            continue
        # Skip very large files (>5MB — probably binary data)
        if filepath.stat().st_size > 5_000_000:
            continue
        files.append(filepath)

    return sorted(files)


def index_codebase(root_dir: str, force: bool = False) -> int:
    """
    Indexa el codebase en LanceDB con soporte INCREMENTAL.

    - Primera vez o force=True: indexa todo.
    - Normal restarts: compares file hashes against the on-disk index.
      Only re-embeds new or modified files. Removes chunks from deleted files.

    Returns: number of new/updated chunks processed.
    """
    global _lance_table, _indexed

    effective_force = force or FORCE_REINDEX

    if _indexed and not effective_force:
        logger.info("[RAG] Codebase already indexed this session. Skipping.")
        return 0

    embedder = _get_embedder()
    if embedder is None:
        return 0

    db, existing_table = _get_lance()
    if db is None:
        return 0

    files = _collect_files(root_dir)
    logger.info("[RAG] %d archivos encontrados en: %s", len(files), root_dir)

    # ── PRIMERA VEZ o FORCE: indexar todo ─────────────────────────────────────
    if existing_table is None or effective_force:
        if existing_table is not None and effective_force:
            db.drop_table(TABLE_NAME)
            logger.info("[RAG] Index anterior eliminado (force reindex)")

        all_chunks = []
        for f in files:
            all_chunks.extend(_chunk_file(f))

        if not all_chunks:
            logger.warning("[RAG] Sin chunks para indexar.")
            return 0

        logger.info("[RAG] Full index: %d chunks de %d archivos...", len(all_chunks), len(files))
        _embed_and_store(db, all_chunks, mode="create")
        _indexed = True
        logger.info("✅ [RAG] Codebase indexado: %d chunks de %d archivos", len(all_chunks), len(files))
        return len(all_chunks)

    # ── INCREMENTAL: solo archivos nuevos o modificados ────────────────────────
    row_count = existing_table.count_rows()
    if row_count == 0:
        # Table exists but empty → full index
        all_chunks = [c for f in files for c in _chunk_file(f)]
        _embed_and_store(db, all_chunks, mode="create")
        _indexed = True
        return len(all_chunks)

    # Obtener hashes de archivos ya indexados
    try:
        existing_rows = existing_table.to_pandas()[["filepath", "file_hash"]].drop_duplicates()
        indexed_hashes = dict(zip(existing_rows["filepath"], existing_rows["file_hash"]))
    except Exception:
        # If pandas fails (new column didn't exist), fallback to full reindex
        logger.warning("[RAG] Could not read file_hash from index → full reindex")
        all_chunks = [c for f in files for c in _chunk_file(f)]
        db.drop_table(TABLE_NAME)
        _embed_and_store(db, all_chunks, mode="create")
        _indexed = True
        return len(all_chunks)

    current_paths = {str(f) for f in files}
    indexed_paths = set(indexed_hashes.keys())

    # Archivos eliminados del filesystem
    deleted = indexed_paths - current_paths
    # Archivos nuevos
    new_files = [f for f in files if str(f) not in indexed_hashes]
    # Modified files (hash changed)
    modified_files = [
        f for f in files
        if str(f) in indexed_hashes
        and _chunk_file(f)  # avoid crash on empty files
        and hashlib.sha256(
            f.read_text(encoding="utf-8", errors="ignore").encode()
        ).hexdigest()[:16] != indexed_hashes[str(f)]
    ]

    changed = new_files + modified_files

    if not changed and not deleted:
        logger.info("✅ [RAG] Index up to date — no files changed (%d chunks on disk)", row_count)
        _lance_table = existing_table
        _indexed = True
        return 0

    logger.info("[RAG] Incremental: +%d nuevos | ~%d modificados | -%d eliminados",
                len(new_files), len(modified_files), len(deleted))

    # Eliminar chunks obsoletos (archivos borrados o modificados)
    paths_to_remove = deleted | {str(f) for f in modified_files}
    if paths_to_remove:
        for p in paths_to_remove:
            try:
                existing_table.delete(f"filepath = '{p}'")
            except Exception as e:
                logger.warning("[RAG] Could not delete chunks from %s: %s", p, e)

    # Embeber y agregar chunks nuevos/actualizados
    new_chunks = [c for f in changed for c in _chunk_file(f)]
    if new_chunks:
        _embed_and_store(db, new_chunks, mode="append")

    _lance_table = _get_lance()[1]  # refrescar referencia
    _indexed = True
    logger.info("✅ [RAG] Incremental completo: +%d chunks nuevos", len(new_chunks))
    return len(new_chunks)


def _embed_and_store(db, chunks: List[Dict], mode: str = "create"):
    """Embed chunks and store them in LanceDB. mode='create' or 'append'."""
    global _lance_table

    embedder = _get_embedder()
    if embedder is None or not chunks:
        return

    t0 = time.time()
    # Qwen3-Embedding: documentos = texto plano (sin prefijo).
    # Solo las queries llevan instruccion -- diseno asimetrico del modelo.
    texts_for_embedding = [
        c["text"][:800] for c in chunks
    ]
    vectors = embedder.encode(
        texts_for_embedding,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    elapsed = time.time() - t0
    logger.info("[RAG] %d embeddings generados en %.1fs (%.0f chunks/s)",
                len(vectors), elapsed, len(vectors) / elapsed if elapsed > 0 else 0)

    records = []
    for i, chunk in enumerate(chunks):
        records.append({
            "vector": vectors[i].tolist(),
            "text": chunk["text"],
            "filepath": chunk["filepath"],
            "filename": chunk["filename"],
            "file_hash": chunk["file_hash"],
            "start_line": chunk["start_line"],
            "end_line": chunk["end_line"],
            "extension": chunk["extension"],
            "chunk_id": chunk["chunk_id"],
        })

    import lancedb
    if mode == "create":
        _lance_table = db.create_table(TABLE_NAME, data=records)
    else:
        _lance_table = db.open_table(TABLE_NAME)
        _lance_table.add(records)


# ── Search and Enrichment ─────────────────────────────────────────────────────

def _estimate_tokens(text: str) -> int:
    """Quick estimation: ~4 chars per token."""
    return len(text) // 4


def search_relevant_context(query: str, top_k: int = TOP_K,
                            relevance_threshold: Optional[float] = None) -> List[Dict]:
    """
    Searches for relevant chunks matching the user's query.

    Args:
        query: text to search.
        top_k: maximum number of chunks to retrieve.
        relevance_threshold: max L2 distance to consider a chunk relevant.
            None → uses module RELEVANCE_THRESHOLD (calibrated for OpenClaw, 1.6).
            Explicit float → override for caller (e.g. sidecar uses 1.0 for stricter filtering).

    Returns: list of dicts with {text, filepath, filename, start_line, score}
    """
    embedder = _get_embedder()
    _, table = _get_lance()

    if embedder is None or table is None:
        return []

    try:
        # Qwen3-Embedding: instruction prefix mejora cross-lingual 1-5%
        query_text = (
            f"Instruct: Retrieve relevant passages to answer the question\n"
            f"Query: {query[:500]}"
        )
        query_vector = embedder.encode(query_text, normalize_embeddings=True).tolist()

        results = table.search(query_vector).limit(top_k).to_list()

        threshold = relevance_threshold if relevance_threshold is not None else RELEVANCE_THRESHOLD

        # ALWAYS log distances for calibration
        distances = [f"{r.get('_distance', -1):.3f}" for r in results]
        logger.info("[RAG] Distancias top-%d: %s (threshold=%.1f)", len(results), distances, threshold)

        filtered = [r for r in results if r.get("_distance", 2.0) < threshold]

        if len(filtered) < len(results):
            logger.info("[RAG] Filtrados %d/%d chunks por baja relevancia",
                        len(results) - len(filtered), len(results))

        return [{
            "text": r["text"],
            "filepath": r["filepath"],
            "filename": r["filename"],
            "start_line": r["start_line"],
            "score": r.get("_distance", 0),
        } for r in filtered]

    except Exception as e:
        logger.error("[RAG] Error in search: %s", e)
        return []


# ── OpenClaw system prefixes that are NOT user questions ──
_OPENCLAW_SYSTEM_PREFIXES = (
    "A new session was started",
    "Session resumed",
    "You are being restarted",
)


def _extract_clean_query(messages: List[Dict]) -> Optional[str]:
    """Extract and clean the user query from messages.

    Handles:
    - content as plain string
    - content as list [{type: "text", text: "..."}] (OpenClaw format)
    - Discards OpenClaw system messages (/new, /reset)
    - Strips Telegram/OpenClaw metadata
    - Discards queries < 20 chars

    Returns: clean query ready for vector search, or None if should skip.
    """
    user_query = ""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, str):
                user_query = content
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        user_query = part.get("text", "")
                        break
            break

    if not user_query:
        return None

    if user_query.lstrip().startswith(_OPENCLAW_SYSTEM_PREFIXES):
        logger.info("[RAG] Skip: OpenClaw system message detected (not a user question)")
        return None

    clean_query = _strip_openclaw_metadata(user_query)

    if len(clean_query) < 20:
        logger.info("[RAG] Short query (%d chars) — skip enrichment", len(clean_query))
        return None

    return clean_query


def enrich_messages(messages: List[Dict],
                    max_context_tokens: int = MAX_CONTEXT_TOKENS,
                    relevance_threshold: Optional[float] = None) -> List[Dict]:
    """
    Main entry point. Receives user messages and returns
    messages enriched with relevant codebase context.

    Injected BEFORE the last user message to preserve the stable token prefix
    (system + tools + history) and maximize KV cache hits.

    Args:
        messages: lista de messages OpenAI-compatible.
        max_context_tokens: max tokens to inject.
        relevance_threshold: override for L2 relevance threshold.
            None → uses module RELEVANCE_THRESHOLD (OpenClaw default: 1.6).
            Explicit float → sidecar passes 1.0 to filter irrelevant chunks.
            DOES NOT AFFECT OpenClaw since OpenClaw does not pass this parameter.
    """
    if not _indexed:
        return messages     # RAG not ready, pass through without enrichment

    clean_query = _extract_clean_query(messages)
    if clean_query is None:
        return messages

    logger.info("[RAG] Clean query for search: %r", clean_query[:150])

    # Buscar contexto relevante con query limpio
    t0 = time.time()
    results = search_relevant_context(clean_query, relevance_threshold=relevance_threshold)
    search_ms = int((time.time() - t0) * 1000)

    if not results:
        logger.info("[RAG] Sin resultados relevantes | query=%r | %dms", clean_query[:80], search_ms)
        return messages

    # Build the injectable context
    # IMPORTANT: truncate chunks that exceed the limit instead of discarding them.
    # With CHUNK_SIZE_LINES=200, each chunk can be ~4000 tokens.
    # If MAX_CONTEXT_TOKENS < chunk size, it discarded EVERYTHING → no injection.
    context_parts = []
    total_tokens = 0
    max_chars = max_context_tokens * 4   # estimate: 4 chars/token

    for r in results:
        remaining_chars = (max_context_tokens - total_tokens) * 4
        if remaining_chars <= 200:        # menos de ~50 tokens disponibles → parar
            break

        # Truncate chunk text if larger than available space
        chunk_text = r["text"][:remaining_chars]
        chunk_tokens = _estimate_tokens(chunk_text)

        header = f"--- {r['filename']} (lines {r['start_line']}+) ---"
        context_parts.append(f"{header}\n{chunk_text}")
        total_tokens += chunk_tokens

        # Log detallado por chunk inyectado
        logger.info("[RAG] chunk[%d]: %s (L%d+) | score=%.3f | ~%d tok",
                    len(context_parts) - 1, r["filename"], r["start_line"],
                    r.get("score", 0), chunk_tokens)

    if not context_parts:
        return messages

    context_block = "\n\n".join(context_parts)
    rag_message = {
        "role": "system",
        "content": (
            "Reference context for the question above — from your "
            "internal memory (curated and verified sources). "
            "Reason over these fragments to build a rich, grounded answer. "
            "If you need to supplement with external searches, make sure to use "
            "only real paths, links, and sources you can verify.\n\n"
            f"{context_block}"
        )
    }

    logger.info("[RAG] Enriched: %d chunks, ~%d tokens injected (%dms search) | query=%r",
                len(context_parts), total_tokens, search_ms, clean_query[:80])

    # Inject BEFORE the last user message to maximize proximity
    # to the user's query. This preserves the stable KV cache prefix
    # (system + tools + full history) and only invalidates the final
    # portion (RAG + user msg), which changes every turn anyway.
    #
    # HISTORICAL NOTE: Injecting AFTER the last user message was tested
    # (at the absolute end), but the 9B model confused RAG chunks with
    # dialogue continuation, causing 19+ tool call loops.
    # PRE-user injection avoids this because role="system" acts as a
    # clear separator before the question.
    #
    # Post-system position (v1) was also discarded: with long history
    # (67+ msgs, ~85K chars), the model systematically ignored the RAG
    # (confirmed via thinking tokens and by Qwen3.5-9B directly).
    enriched = list(messages)  # copy to avoid mutating the original

    # Find the index of the last user message
    last_user_idx = None
    for i in range(len(enriched) - 1, -1, -1):
        if enriched[i].get("role") == "user":
            last_user_idx = i
            break

    if last_user_idx is not None:
        enriched.insert(last_user_idx, rag_message)
    else:
        # Fallback: si no hay user message, inyectar al final
        enriched.append(rag_message)

    return enriched


def enrich_messages_with_metadata(messages: List[Dict],
                                  max_context_tokens: int = MAX_CONTEXT_TOKENS,
                                  relevance_threshold: Optional[float] = None) -> Tuple[List[Dict], Dict]:
    """
    Like enrich_messages(), but also returns RAG search metadata for cascade routing.

    Returns:
        (enriched_messages, metadata) where metadata contains:
        - best_score: float — L2 distance of best chunk (lower=better, 999.0 if none found)
        - chunks_found: int — number of relevant chunks injected
        - query_used: str — the cleaned query that was searched
        - search_ms: int — milliseconds spent on vector search
    """
    _no_rag_meta = {"best_score": 999.0, "chunks_found": 0, "query_used": "", "search_ms": 0}

    if not _indexed:
        return messages, _no_rag_meta

    clean_query = _extract_clean_query(messages)
    if clean_query is None:
        return messages, _no_rag_meta

    # Search with score tracking
    t0 = time.time()
    results = search_relevant_context(clean_query, relevance_threshold=relevance_threshold)
    search_ms = int((time.time() - t0) * 1000)

    best_score = min((r.get("score", 999.0) for r in results), default=999.0)
    meta = {
        "best_score": best_score,
        "chunks_found": len(results),
        "query_used": clean_query[:150],
        "search_ms": search_ms,
    }

    if not results:
        return messages, meta

    # Build enriched messages (reuse same injection logic)
    enriched = enrich_messages(messages, max_context_tokens, relevance_threshold)
    meta["chunks_found"] = len(results)
    return enriched, meta


def _strip_openclaw_metadata(raw_query: str) -> str:
    """Extract the real user question by stripping OpenClaw/Telegram wrappers.

    OpenClaw wraps each message with JSON metadata like:
        Conversation info (untrusted metadata):
        ```json
        { "chat_id": "telegram:8331738812", ... }
        ```
        Sender (untrusted metadata):
        ```json
        { "label": "Diego Cassisi", ... }
        ```
        [Sat 2026-04-18 22:09 GMT-3] <actual question here>

    If the pattern is not detected, returns the original query unchanged.
    """
    import re

    # Pattern: everything up to last metadata block close + optional timestamp
    # Find the last ``` that closes a metadata JSON block
    # Luego opcionalmente un timestamp [Mon 2026-04-18 ...] antes de la pregunta real
    parts = raw_query.split("```")

    if len(parts) >= 3:
        # Hay al menos un bloque ```json ... ```.
        # The real question is after the last ```
        after_last_fence = parts[-1].strip()

        # Eliminar timestamp tipo [Sat 2026-04-18 22:09 GMT-3]
        after_last_fence = re.sub(
            r'^\[.*?GMT[+-]?\d*\]\s*', '', after_last_fence
        ).strip()

        if len(after_last_fence) >= 10:
            return after_last_fence

    # Fallback: buscar el timestamp directamente y tomar lo que sigue
    ts_match = re.search(r'\[.*?GMT[+-]?\d*\]\s*', raw_query)
    if ts_match:
        after_ts = raw_query[ts_match.end():].strip()
        if len(after_ts) >= 10:
            return after_ts

    # No metadata detected — return original
    return raw_query.strip()


# ── Compressor (reduces long history) ─────────────────────────────────────────

_chat_table = None
_seen_chat_hashes = set()


def _estimate_tokens(text: str) -> int:
    """Quick estimation: ~4 chars per token."""
    return len(text) // 4


def _load_reranker():
    """Load reranker cross-encoder on CPU. Lazy init."""
    global _reranker
    if _reranker is not None:
        return _reranker
    if not COMPRESSOR_RERANKER_ENABLED:
        return None
    try:
        from sentence_transformers import CrossEncoder
        logger.info("[COMPRESSOR] Loading Reranker: %s (CPU)", COMPRESSOR_RERANKER_MODEL)
        _reranker = CrossEncoder(COMPRESSOR_RERANKER_MODEL, device="cpu")
        logger.info("[COMPRESSOR] Reranker ready on CPU")
        return _reranker
    except Exception as e:
        logger.error("[COMPRESSOR] Could not load Reranker: %s", e)
        return None


def _load_llmlingua():
    """Load LLMLingua-2 on CPU. Lazy init."""
    global _llmlingua
    if _llmlingua is not None:
        return _llmlingua
    if not COMPRESSOR_LLMLINGUA_ENABLED:
        return None
    try:
        from llmlingua import PromptCompressor
        logger.info("[COMPRESSOR] Loading LLMLingua-2: %s (CPU)", COMPRESSOR_LLMLINGUA_MODEL)
        _llmlingua = PromptCompressor(
            model_name=COMPRESSOR_LLMLINGUA_MODEL,
            use_llmlingua2=True,
            device_map="cpu"
        )
        logger.info("[COMPRESSOR] LLMLingua-2 ready on CPU")
        return _llmlingua
    except Exception as e:
        logger.error("[COMPRESSOR] Could not load LLMLingua-2: %s", e)
        return None


def _rerank_chunks(query: str, chunks: List[str], top_n: int = COMPRESSOR_RERANKER_TOP_N) -> List[str]:
    """Reordena chunks por relevancia usando cross-encoder. CPU.
    
    NOTA: ms-marco-MiniLM-L-6-v2 tiene max_seq_length=512 tokens.
    Los chunks se truncan a ~480 tokens (1920 chars) SOLO para scoring.
    El texto completo se preserva en la salida.
    """
    reranker = _load_reranker()
    if reranker is None or not chunks:
        return chunks[:top_n]
    try:
        t0 = time.time()
        # Truncar chunks para el reranker (max_seq_length=512 tokens ≈ 2048 chars).
        # Reservar ~32 tokens para el query, dejar ~480 para el chunk.
        _RERANKER_MAX_CHARS = 1920  # ~480 tokens × 4 chars/token
        pairs = [[query[:400], chunk[:_RERANKER_MAX_CHARS]] for chunk in chunks]
        scores = reranker.predict(pairs)
        ranked = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)
        result = [c for _, c in ranked[:top_n]]
        elapsed = int((time.time() - t0) * 1000)
        logger.info("[COMPRESSOR] Reranker: %d→%d chunks (%dms)", len(chunks), len(result), elapsed)
        return result
    except Exception as e:
        logger.error("[COMPRESSOR] Reranker failed: %s", e)
        return chunks[:top_n]


def _compress_with_llmlingua(text: str, ratio: float = COMPRESSOR_LLMLINGUA_RATIO) -> str:
    """Fine-grained text compression using LLMLingua-2 on CPU."""
    lingua = _load_llmlingua()
    if lingua is None or not text.strip():
        return text
    try:
        t0 = time.time()
        result = lingua.compress_prompt(text, rate=ratio, force_tokens=["\n"])
        compressed = result.get("compressed_prompt", text)
        elapsed = int((time.time() - t0) * 1000)
        orig_words = len(text.split())
        comp_words = len(compressed.split())
        logger.info("[COMPRESSOR] LLMLingua-2: %d→%d words (-%.0f%%, %dms)",
                    orig_words, comp_words,
                    100 * (1 - comp_words / max(orig_words, 1)), elapsed)
        return compressed
    except Exception as e:
        logger.error("[COMPRESSOR] LLMLingua-2 failed: %s", e)
        return text


def _chunk_chat_messages(messages: List[Dict], chunk_size: int = CHUNK_SIZE_TOKENS) -> List[Dict]:
    """Split chat history into coherent semantic blocks.
    Filters non-conversational roles (tool) and strips OpenClaw noise."""
    chunks = []
    current_chunk = []
    current_tokens = 0

    for msg in messages:
        # Filtro 1: Ignorar tool results
        role = msg.get("role", "")
        if role in _SKIP_ROLES:
            continue

        content = msg.get("content", "")
        if isinstance(content, list):
            content = "\n".join(
                b.get("text", "") for b in content
                if isinstance(b, dict) and b.get("type") == "text"
            )
        content = _sanitize_content_for_rag(content)

        # Filter 2: Skip if empty after sanitizing
        if not content.strip():
            continue

        clean_msg = {**msg, "content": content}
        msg_tokens = _estimate_tokens(str(clean_msg))

        if current_tokens + msg_tokens > chunk_size and current_chunk:
            chunks.append({
                "text": json.dumps(current_chunk, ensure_ascii=False),
                "role": current_chunk[0].get("role", "history"),
                "tokens": current_tokens,
                "count": len(current_chunk),
            })
            current_chunk = []
            current_tokens = 0

        current_chunk.append(clean_msg)
        current_tokens += msg_tokens

    if current_chunk:
        chunks.append({
            "text": json.dumps(current_chunk, ensure_ascii=False),
            "role": current_chunk[0].get("role", "history"),
            "tokens": current_tokens,
            "count": len(current_chunk),
        })

    return chunks


def _index_chat_chunks(chunks: List[Dict], session_key: str):
    """Index history blocks in LanceDB (chat_sessions table)."""
    global _chat_table, _seen_chat_hashes

    embedder = _get_embedder()
    db, _ = _get_lance()
    if embedder is None or db is None:
        return

    # Filtrar chunks ya vistos
    new_chunks = []
    for c in chunks:
        h = hashlib.sha256(c["text"].encode()).hexdigest()[:16]
        uid = f"{session_key}_{h}"
        if uid not in _seen_chat_hashes:
            new_chunks.append((c, uid))

    if not new_chunks:
        return

    # Embeddings en CPU
    texts = [f"search_document: {c['text'][:500]}" for c, _ in new_chunks]
    vectors = embedder.encode(texts, normalize_embeddings=True)

    records = []
    for i, (chunk, uid) in enumerate(new_chunks):
        records.append({
            "vector": vectors[i].tolist(),
            "id": uid,
            "session_key": session_key,
            "text": chunk["text"],
            "role": chunk["role"],
            "tokens": chunk["tokens"],
        })
        _seen_chat_hashes.add(uid)

    import lancedb
    if _chat_table is None:
        if CHAT_TABLE_NAME in db.table_names():
            _chat_table = db.open_table(CHAT_TABLE_NAME)
            _chat_table.add(records)
        else:
            _chat_table = db.create_table(CHAT_TABLE_NAME, data=records)
    else:
        _chat_table.add(records)

    logger.info("[COMPRESSOR] Indexed %d history blocks", len(records))


def _retrieve_chat_context(query: str, session_key: str) -> List[str]:
    """Search for history blocks relevant to the current query."""
    global _chat_table

    embedder = _get_embedder()
    db, _ = _get_lance()
    if embedder is None or db is None:
        return []

    if _chat_table is None:
        if CHAT_TABLE_NAME in db.table_names():
            _chat_table = db.open_table(CHAT_TABLE_NAME)
        else:
            return []

    if _chat_table.count_rows() == 0:
        return []

    query_vec = embedder.encode(
        f"search_query: {query[:500]}", normalize_embeddings=True
    ).tolist()

    try:
        results = _chat_table.search(query_vec) \
            .where(f"session_key = '{session_key}'") \
            .limit(COMPRESSOR_TOP_K) \
            .to_list()
        return [r["text"] for r in results]

    except Exception as e:
        # If error is due to vector dimension mismatch (embedding model change),
        # drop the stale table and continue without history.
        # The table will be automatically recreated in the next session.
        err_str = str(e)
        if "FixedSizeListType" in err_str or "ListType" in err_str or "expected size" in err_str:
            logger.warning(
                "[COMPRESSOR] ⚠️ Vector dimension mismatch detected. "
                "Table '%s' was created with a different embedding model. "
                "Dropping stale table → will recreate with current model.",
                CHAT_TABLE_NAME,
            )
            try:
                db.drop_table(CHAT_TABLE_NAME)
                _chat_table = None
                logger.info("[COMPRESSOR] ✅ Table '%s' dropped. Clean history.", CHAT_TABLE_NAME)
            except Exception as drop_err:
                logger.error("[COMPRESSOR] Could not drop stale table: %s", drop_err)
        else:
            logger.error("[COMPRESSOR] Error in history search: %s", e)
        return []


def compress_messages(messages: List[Dict],
                      session_key: str = "default") -> List[Dict]:
    """
    Compress long conversation history.
    Keeps system + last COMPRESSOR_GUARD messages intact.
    Old messages are indexed in LanceDB and replaced by
    a semantically relevant summary for the current query.

    Inspired by OpenClaw RAG Proxy compress_prompt().
    """
    if not COMPRESSOR_ENABLED:
        return messages

    total_tokens = sum(_estimate_tokens(json.dumps(m)) for m in messages)

    if total_tokens < COMPRESSION_THRESHOLD:
        return messages

    logger.info("[COMPRESSOR] History exceeds threshold (%d tokens). Compressing...", total_tokens)

    # Separar system de non-system
    system_msgs = [m for m in messages if m.get("role") == "system"]
    non_system = [m for m in messages if m.get("role") != "system"]

    if len(non_system) <= COMPRESSOR_GUARD:
        # Aun sin comprimir, truncar assistant messages largos en el historial
        MAX_ASSISTANT_TOKENS = 500
        truncated = []
        for m in messages:
            if m.get("role") == "assistant":
                content = m.get("content", "")
                if _estimate_tokens(content) > MAX_ASSISTANT_TOKENS:
                    # Save start + end of message (context + conclusion)
                    cut_chars = MAX_ASSISTANT_TOKENS * 4  # ~4 chars/token
                    head = content[:cut_chars // 2]
                    tail = content[-(cut_chars // 2):]
                    m = {**m, "content": head + "\n\n[... response truncated for efficiency ...]\n\n" + tail}
                    logger.info("[COMPRESSOR] Truncated assistant message: %d → ~%d tokens",
                                _estimate_tokens(content), MAX_ASSISTANT_TOKENS)
            truncated.append(m)
        return truncated

    # Guard: protect last N messages
    # Ensure the cut doesn't leave an orphan assistant at the start
    cut_point = len(non_system) - COMPRESSOR_GUARD
    while cut_point > 0 and non_system[cut_point].get("role") not in ("user",):
        cut_point -= 1

    recent_msgs = non_system[cut_point:]
    old_msgs = non_system[:cut_point]

    if not old_msgs:
        return messages

    # Indexar historial viejo
    chunks = _chunk_chat_messages(old_msgs)
    _index_chat_chunks(chunks, session_key)

    # Recuperar contexto relevante al query actual
    last_user = ""
    for m in reversed(recent_msgs):
        if m.get("role") == "user":
            last_user = str(m.get("content", ""))
            break

    raw_chunks = _retrieve_chat_context(last_user, session_key)

    if not raw_chunks:
        return messages

    # ── Reranker: poda gruesa (COMPRESSOR_TOP_K → RERANKER_TOP_N) ──
    reranked_chunks = _rerank_chunks(last_user, raw_chunks)

    # ── Construir texto de contexto ──
    context_raw = ""
    for txt in reranked_chunks:
        try:
            chunk_msgs = json.loads(txt)
            for m in chunk_msgs:
                role = m.get("role", "unknown").upper()
                content = m.get("content", "")
                if isinstance(content, str) and content.strip():
                    context_raw += f"[{role}]: {content}\n\n"
        except Exception:
            continue

    if context_raw:
        # ── LLMLingua-2: fine-grained compression (instead of truncating to 300 chars) ──
        context_compressed = _compress_with_llmlingua(context_raw)

        system_msgs.append({
            "role": "system",
            "content": (
                "Relevant historical context (compressed):\n"
                f"{context_compressed}"
            )
        })

    compressed = system_msgs + recent_msgs
    new_tokens = sum(_estimate_tokens(json.dumps(m)) for m in compressed)

    logger.info("[COMPRESSOR] Compression: %d → %d tokens (-%.0f%%)",
                total_tokens, new_tokens,
                100 * (1 - new_tokens / total_tokens) if total_tokens > 0 else 0)

    return compressed


# ── Public API for server.py ──────────────────────────────────────────────────

def init(codebase_root: str = None):
    """
    Initialize the RAG enricher. Call at server startup.

    Args:
        codebase_root: root directory of the codebase to index.
                       Default: ../  (project root)
    """
    if codebase_root is None:
        codebase_root = str(pathlib.Path(__file__).parent.parent)

    logger.info("🔧 [RAG] Initializing with root: %s", codebase_root)
    logger.info("🔧 [RAG] Switches: ENRICHER=%s COMPRESSOR=%s", ENRICHER_ENABLED, COMPRESSOR_ENABLED)
    n = index_codebase(codebase_root)
    logger.info("🔧 [RAG] Init complete. %d chunks indexed.", n)
    return n
