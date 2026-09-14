"""
[AI_DIRECTIVE]
ROL: Punto de entrada para el módulo aislado de Multi-Token Prediction (MTP) en SuperMLX.
OBJETIVO: Exportar las funciones públicas de inyección de MTP y shimming de arquitectura sin efectos secundarios al importar.
ENTRADAS: Ninguna al importar.
SALIDAS: Símbolos públicos (install_qwen3_5_mtp_trunk_shim, inject_qwen3_5_mtp_support, validate_qwen3_5_mtp_support).
REGLAS INVIOLABLES:
- Prohibido modificar módulos de site-packages/mlx_lm en disco.
- Obligatorio mantener aislamiento total: si ENABLE_MTP=false, este módulo no debe interferir con la inferencia base.
SSoT: supermlx.mtp.qwen_mtp_shim
"""

from .qwen_mtp_shim import (
    install_qwen3_5_mtp_trunk_shim,
    inject_qwen3_5_mtp_support,
    validate_qwen3_5_mtp_support,
)
from .speculative_engine import MTPSpeculativeEngine, stream_generate_mtp

__all__ = [
    "install_qwen3_5_mtp_trunk_shim",
    "inject_qwen3_5_mtp_support",
    "validate_qwen3_5_mtp_support",
    "MTPSpeculativeEngine",
    "stream_generate_mtp",
]
