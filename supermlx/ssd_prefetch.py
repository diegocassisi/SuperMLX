# SPDX-License-Identifier: MIT
"""
[AI_DIRECTIVE]
ROL: Async SSD prefetch via Darwin F_RDADVISE for SuperMLX expert loading.
OBJETIVO: Eliminate page fault serialization (QD=1→QD=32) by issuing
    read-ahead hints to macOS kernel before mmap access.
ENTRADAS: File descriptors from SafetensorsMap, byte ranges from _index.
SALIDAS: Pre-warmed page cache pages — zero-latency mmap access on subsequent reads.
REGLAS INVIOLABLES:
- NEVER overlap prefetch with GPU compute (memory controller contention).
- F_RDADVISE only — posix_madvise(WILLNEED) is a no-op on Darwin.
- Fire-and-forget: prefetch failure must never break loading.
- Thread pool workers ≤ 8 (avoid SSD queue saturation).
SSoT: tkhr-sait/ds4 F_RDADVISE implementation (ds4 #172).
    Flash-MoE unified memory findings (no SSD↔GPU overlap).
"""

import fcntl
import logging
import struct
import sys
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Darwin fcntl.h: F_RDADVISE = 44
# struct radvisory { off_t ra_offset (8B); int ra_count (4B); } = 12 bytes
_F_RDADVISE = 44
_RADVISORY_FMT = "qi"  # off_t (long long, 8B) + int (4B)
_AVAILABLE = sys.platform == "darwin"

# Thread pool — small to avoid SSD queue saturation
_pool: Optional[ThreadPoolExecutor] = None
_MAX_WORKERS = 4

# Stats for telemetry
_stats = {"issued": 0, "failed": 0}


def _ensure_pool() -> ThreadPoolExecutor:
    """Lazy-init the prefetch thread pool."""
    global _pool
    if _pool is None:
        _pool = ThreadPoolExecutor(
            max_workers=_MAX_WORKERS,
            thread_name_prefix="ssd_prefetch",
        )
    return _pool


def _rdadvise(fd: int, offset: int, count: int) -> bool:
    """Issue a single F_RDADVISE hint. Returns True on success."""
    try:
        buf = struct.pack(_RADVISORY_FMT, offset, count)
        fcntl.fcntl(fd, _F_RDADVISE, buf)
        _stats["issued"] += 1
        return True
    except OSError:
        _stats["failed"] += 1
        return False


def prefetch_ranges(fd: int, ranges: List[Tuple[int, int]]) -> int:
    """Issue F_RDADVISE for multiple byte ranges on a file descriptor.

    Each range is (byte_offset, byte_length). The kernel starts
    read-ahead I/O immediately — pages will be in the page cache
    by the time the subsequent mmap access touches them.

    Priority is encoded by repeat count (per tkhr-sait/ds4):
    calling this twice on the same range raises I/O priority.

    Args:
        fd: File descriptor number (from fileno()).
        ranges: List of (offset, length) tuples.

    Returns:
        Number of successfully issued hints.
    """
    issued = 0
    for offset, length in ranges:
        if _rdadvise(fd, offset, length):
            issued += 1
    return issued


def prefetch_ranges_async(fd: int, ranges: List[Tuple[int, int]]) -> None:
    """Fire-and-forget async prefetch via thread pool.

    Non-blocking: submits to a background thread and returns immediately.
    Failures are silently counted in _stats — never raises.
    """
    if not ranges or not _AVAILABLE:
        return
    pool = _ensure_pool()
    pool.submit(prefetch_ranges, fd, ranges)


def compute_expert_ranges(
    index: Dict[str, tuple],
    key: str,
    expert_ids: List[int],
) -> Tuple[Optional[str], List[Tuple[int, int]]]:
    """Compute byte ranges for specific expert rows in a stacked tensor.

    Args:
        index: SafetensorsMap._index dict.
        key: Tensor key (e.g. "model.layers.0.mlp.switch_mlp.gate_proj.weight").
        expert_ids: Global expert IDs to compute ranges for.

    Returns:
        (shard_path, [(offset, length), ...]) or (None, []) if key missing.
    """
    if key not in index:
        return None, []

    path, np_dt, mx_dt, shape, start, length = index[key]
    row_bytes = length // shape[0]

    ranges = [(start + int(eid) * row_bytes, row_bytes) for eid in expert_ids]
    return path, ranges


def prefetch_experts(
    index: Dict[str, tuple],
    fds: Dict[str, object],
    key_base: str,
    expert_ids: List[int],
    proj_names: Tuple[str, ...] = ("gate_proj", "up_proj", "down_proj"),
) -> None:
    """Prefetch all projections for given experts in a layer.

    Computes byte ranges from SafetensorsMap._index and issues async
    F_RDADVISE hints per shard file descriptor. Non-blocking.

    Args:
        index: SafetensorsMap._index
        fds: SafetensorsMap._fds (path → file object)
        key_base: e.g. "model.layers.0.mlp.switch_mlp"
        expert_ids: Expert IDs to prefetch.
        proj_names: Projection names to prefetch.
    """
    if not _AVAILABLE or not expert_ids:
        return

    # Group ranges by shard path to minimize fd lookups
    path_ranges: Dict[str, List[Tuple[int, int]]] = {}

    for proj_name in proj_names:
        for suffix in ("weight", "scales", "biases"):
            key = f"{key_base}.{proj_name}.{suffix}"
            path, ranges = compute_expert_ranges(index, key, expert_ids)
            if path is not None and ranges:
                path_ranges.setdefault(path, []).extend(ranges)

    for path, ranges in path_ranges.items():
        fd_obj = fds.get(path)
        if fd_obj is not None:
            prefetch_ranges_async(fd_obj.fileno(), ranges)


def get_stats() -> Dict[str, int]:
    """Return prefetch telemetry."""
    return dict(_stats)


def shutdown() -> None:
    """Shutdown the prefetch thread pool."""
    global _pool
    if _pool is not None:
        _pool.shutdown(wait=False)
        _pool = None
