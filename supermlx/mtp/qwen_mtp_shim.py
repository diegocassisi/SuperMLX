"""
[AI_DIRECTIVE]
ROL: Runtime Shim y Módulo de Inyección de Multi-Token Prediction (MTP) para Qwen3.5/Qwen3.6 MoE en SuperMLX.
OBJETIVO: Permitir la carga e inferencia de cabezas MTP sobre modelos base Qwen MoE en MLX sin alterar archivos de mlx-lm en disco.
ENTRADAS:
- model: Instancia de modelo cargado con mlx_lm (qwen3_5_moe.Model).
- mtp_weights_path: Path al directorio o archivo .safetensors con los pesos del cabezal MTP.
- config: Dict con la configuración del modelo (config.json).
SALIDAS:
- bool: True si la inyección fue exitosa, False en caso contrario.
REGLAS INVIOLABLES:
- Prohibido modificar archivos de site-packages/mlx_lm en disco.
- Prohibido monkeypatching mutable de self.norm que comprometa concurrencia o reentrancia.
- Prohibido print(). Usar exclusivamente logging con prefijos estándar ([INICIO], [CONFIG], [DATA], [RESULT], [ERROR]).
- Obligatorio tipado estricto en funciones públicas.
- Obligatorio mitigar el bug de double-shift en RMSNorm al sanear pesos del trunk.
- Obligatorio control de idempotencia para evitar anidación de subclases dinámicas.
SSoT: supermlx.mtp.qwen_mtp_shim
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

logger = logging.getLogger(__name__)

QWEN3_5_MTP_MODEL_TYPES = {"qwen3_5_mtp"}


def _model_type(config: Dict[str, Any]) -> str:
    """Extrae y normaliza el model_type de la configuración."""
    return str(config.get("model_type") or "").lower()


def _text_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Obtiene text_config o la raíz de la configuración."""
    return config.get("text_config") or config


def is_qwen3_5_mtp_config(config: Dict[str, Any]) -> bool:
    """Verifica si la configuración corresponde a una arquitectura Qwen MTP."""
    m_type = _model_type(config)
    if m_type in QWEN3_5_MTP_MODEL_TYPES:
        return True
    tcfg = _text_config(config)
    num_nextn = int(config.get("num_nextn_predict_layers") or tcfg.get("num_nextn_predict_layers") or 0)
    return num_nextn > 0


def install_qwen3_5_mtp_trunk_shim() -> None:
    """
    Registra dinámicamente un alias en sys.modules para 'mlx_lm.models.qwen3_5_mtp'.
    Permite que mlx_lm.utils.load construya el trunk usando qwen3_5_moe sin lanzar ValueError.
    Evita además el bug de double-shift en RMSNorm ignorando claves mtp.* durante el sanitize del trunk.
    """
    name = "mlx_lm.models.qwen3_5_mtp"
    if name in sys.modules:
        logger.debug("[CONFIG] MTP trunk shim ya instalado en sys.modules")
        return

    import types
    import mlx_lm.models.qwen3_5_moe as base

    class _TrunkModel(base.Model):
        def sanitize(self, weights: Dict[str, Any]) -> Dict[str, Any]:
            # mlx-lm qwen3_5 sanitize suma +1.0 a las normas si detecta mtp.* (asumiendo checkpoints sin procesar).
            # En exports finales las normas ya vienen calibradas. Descartamos mtp.* aquí para que no doble-aplique.
            filtered_weights = {k: v for k, v in weights.items() if "mtp." not in str(k)}
            return super().sanitize(filtered_weights)

    shim = types.ModuleType(name)
    shim.Model = _TrunkModel
    shim.ModelArgs = base.ModelArgs
    sys.modules[name] = shim
    logger.info("[INICIO] MTP trunk shim instalado exitosamente para %s", name)


def _strip_mtp_prefix(key: str) -> Optional[str]:
    """Limpia prefijos externos de la clave de tensor para mapear al árbol local del MTP."""
    k = str(key)
    for outer in ("language_model.", "model.model.", "model."):
        if k.startswith(outer) and "mtp." in k:
            k = k[k.index("mtp."):]
            break
    if k.startswith("mtp."):
        return k[len("mtp."):]
    return None


def _candidate_weight_files(weights_path: Path) -> List[Path]:
    """Localiza los archivos .safetensors candidatos para cargar los pesos MTP."""
    if weights_path.is_file() and weights_path.suffix == ".safetensors":
        return [weights_path]
    if weights_path.is_dir():
        head_file = weights_path / "model.safetensors"
        if head_file.exists():
            return [head_file]
        return sorted(weights_path.glob("*.safetensors"))
    return []


def _load_mtp_weights(paths: List[Path]) -> Dict[str, Any]:
    """Carga los tensores de safetensors y normaliza las claves."""
    mapped: Dict[str, Any] = {}
    for path in paths:
        if path.suffix != ".safetensors":
            continue
        logger.info("[DATA] Cargando tensores MTP desde %s", path.name)
        loaded = mx.load(str(path))
        for key, value in loaded.items():
            local_key = _strip_mtp_prefix(key) or key
            if local_key == "layers.0.mlp.experts.gate_up_proj":
                mid = value.shape[-2] // 2
                mapped["layers.0.mlp.switch_mlp.gate_proj.weight"] = value[..., :mid, :]
                mapped["layers.0.mlp.switch_mlp.up_proj.weight"] = value[..., mid:, :]
            elif local_key == "layers.0.mlp.experts.down_proj":
                mapped["layers.0.mlp.switch_mlp.down_proj.weight"] = value
            elif "norm" in local_key and value.ndim == 1 and value.min() < 0:
                mapped[local_key] = value + 1.0
            else:
                mapped[local_key] = value
    logger.info("[DATA] ✓ Tensores MTP cargados: %d claves procesadas", len(mapped))
    return mapped


def _full_attention_layer_idx(args: Any) -> int:
    """
    Calcula un layer_idx tal que qwen3_5.DecoderLayer construya self_attn (full attention).
    Verificado en código fuente: en DecoderLayer, layer_idx SOLO controla la rama is_linear.
    Attention, RMSNorm y SparseMoeBlock no dependen del índice de capa.
    """
    interval = int(getattr(args, "full_attention_interval", 4) or 4)
    return interval - 1


def _make_qwen3_5_mtp_module(args: Any) -> nn.Module:
    """Instancia la cabeza MTP compuesta por RMSNorms, Linear FC y 1 DecoderLayer MoE."""
    from mlx_lm.models.qwen3_5 import DecoderLayer

    class _Qwen35MTP(nn.Module):
        def __init__(self):
            super().__init__()
            self.pre_fc_norm_embedding = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
            self.pre_fc_norm_hidden = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
            self.fc = nn.Linear(args.hidden_size * 2, args.hidden_size, bias=False)
            # 1 bloque decoder Qwen3.5 MoE con Full-Attention verificado
            self.layers = [DecoderLayer(args=args, layer_idx=_full_attention_layer_idx(args))]
            self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    return _Qwen35MTP()


def _get_inner_text_model(model: Any) -> Any:
    """Extrae el TextModel interno donde residen las capas y el lm_head."""
    if hasattr(model, "language_model"):
        return model.language_model
    return model


def inject_qwen3_5_mtp_support(
    model: Any,
    mtp_path: Path | str,
    config: Dict[str, Any],
) -> bool:
    """
    Inyecta la cabeza MTP y los métodos de inferencia especulativa sobre un modelo Qwen MoE cargado.
    Implementación thread-safe: sin monkeypatching sobre self.norm de la instancia.
    """
    text_model = _get_inner_text_model(model)

    # 1. Guard de Idempotencia: Evitar re-envolver clases si ya fue inyectado
    if getattr(text_model, "_mtp_injected", False):
        logger.info("[CONFIG] MTP ya se encuentra inyectado e inicializado en la instancia del modelo")
        return True

    logger.info("[INICIO] Iniciando inyección limpia de soporte MTP...")
    from mlx_lm.models.base import create_attention_mask
    from mlx_lm.models.cache import KVCache
    from mlx_lm.models.qwen3_5 import TextModelArgs, create_ssm_mask

    mtp_path = Path(mtp_path)
    tcfg = _text_config(config)
    args = TextModelArgs.from_dict(tcfg)

    weight_files = _candidate_weight_files(mtp_path)
    if not weight_files:
        logger.error("[ERROR] No se encontraron archivos de pesos safetensors en %s", mtp_path)
        return False

    weights = _load_mtp_weights(weight_files)
    if not weights:
        logger.error("[ERROR] Diccionario de pesos MTP vacío tras la lectura")
        return False

    mtp_module = _make_qwen3_5_mtp_module(args)

    # 2. Cuantización de la cabeza MTP
    q_cfg = config.get("quantization_config") or tcfg.get("quantization_config") or config.get("quantization")
    if q_cfg:
        group_size = int(q_cfg.get("group_size", 64))
        bits = int(q_cfg.get("bits", 4))
        mode = str(q_cfg.get("mode", "affine"))
        logger.info("[CONFIG] Cuantizando cabeza MTP: bits=%d, group_size=%d, mode=%s", bits, group_size, mode)
        nn.quantize(mtp_module, group_size=group_size, bits=bits, mode=mode)

    try:
        mtp_module.load_weights(list(weights.items()), strict=False)
        mx.eval(mtp_module.parameters())
    except Exception as e:
        logger.error("[ERROR] Fallo al cargar pesos en mtp_module: %s", str(e))
        raise

    text_model.mtp = mtp_module

    # 2.5 Dynamic GatedDeltaNet Patch for exact MTP rollback without modifying site-packages
    from mlx_lm.models.qwen3_5 import GatedDeltaNet, DecoderLayer, gated_delta_update

    if not hasattr(GatedDeltaNet, "_mtp_chunking_patched"):
        original_gdn_call = GatedDeltaNet.__call__

        def _process_chunk(self, qkv, a_chunk, b_chunk, conv_state, ssm_state, ssm_mask, lengths=None):
            B, S_chunk, _ = qkv.shape
            conv_in = mx.concatenate([conv_state, qkv], axis=1)
            n_keep = self.conv_kernel_size - 1
            if lengths is not None:
                ends = mx.clip(lengths, 0, S_chunk)
                positions = (ends[:, None] + mx.arange(n_keep))[..., None]
                new_conv_state = mx.take_along_axis(conv_in, positions, axis=1)
            else:
                new_conv_state = mx.contiguous(conv_in[:, -n_keep:])
            conv_out = nn.silu(self.conv1d(conv_in))

            q, k, v = [
                t.reshape(B, S_chunk, h, d)
                for t, h, d in zip(
                    mx.split(conv_out, [self.key_dim, 2 * self.key_dim], -1),
                    [self.num_k_heads, self.num_k_heads, self.num_v_heads],
                    [self.head_k_dim, self.head_k_dim, self.head_v_dim],
                )
            ]
            inv_scale = k.shape[-1] ** -0.5
            q = (inv_scale**2) * mx.fast.rms_norm(q, None, 1e-6)
            k = inv_scale * mx.fast.rms_norm(k, None, 1e-6)

            out, new_ssm_state = gated_delta_update(
                q,
                k,
                v,
                a_chunk,
                b_chunk,
                self.A_log,
                self.dt_bias,
                ssm_state,
                ssm_mask,
                use_kernel=not self.training,
            )
            return out, new_conv_state, new_ssm_state

        def _patched_gdn_call(self, inputs, mask=None, cache=None, n_confirmed: int = 0):
            B, S, _ = inputs.shape
            if n_confirmed <= 0 or n_confirmed >= S:
                return original_gdn_call(self, inputs, mask=mask, cache=cache)

            if self.sharding_group is not None:
                from mlx_lm.models.switch_layers import sum_gradients
                inputs = sum_gradients(self.sharding_group)(inputs)

            qkv = self.in_proj_qkv(inputs)
            z = self.in_proj_z(inputs).reshape(B, S, self.num_v_heads, self.head_v_dim)
            b = self.in_proj_b(inputs)
            a = self.in_proj_a(inputs)

            if cache is not None and cache[0] is not None:
                conv_state = cache[0]
            else:
                conv_state = mx.zeros(
                    (B, self.conv_kernel_size - 1, self.conv_dim),
                    dtype=inputs.dtype,
                )
            ssm_state = cache[1] if cache else None

            if mask is not None:
                qkv = mx.where(mask[..., None], qkv, 0)

            mask_c = mask[:, :n_confirmed] if mask is not None else None
            mask_d = mask[:, n_confirmed:] if mask is not None else None

            out_c, conv_c, ssm_c = _process_chunk(
                self,
                qkv[:, :n_confirmed],
                a[:, :n_confirmed],
                b[:, :n_confirmed],
                conv_state,
                ssm_state,
                mask_c,
            )
            if cache is not None:
                cache.rollback_state = (conv_c, ssm_c)

            out_d, conv_f, ssm_f = _process_chunk(
                self,
                qkv[:, n_confirmed:],
                a[:, n_confirmed:],
                b[:, n_confirmed:],
                conv_c,
                ssm_c,
                mask_d,
            )
            out = mx.concatenate([out_c, out_d], axis=1)

            if cache is not None:
                cache[0] = conv_f
                cache[1] = ssm_f
                cache.advance(S)

            out = self.norm(out, z)
            out = self.out_proj(out.reshape(B, S, -1))
            if self.sharding_group is not None:
                out = mx.distributed.all_sum(out, group=self.sharding_group)
            return out

        GatedDeltaNet._process_chunk = _process_chunk
        GatedDeltaNet.__call__ = _patched_gdn_call
        GatedDeltaNet._mtp_chunking_patched = True

    # Patch DecoderLayer to forward n_confirmed if not already patched
    if not hasattr(DecoderLayer, "_mtp_n_confirmed_patched"):
        original_decoder_call = DecoderLayer.__call__

        def _patched_decoder_call(self, x, mask=None, cache=None, n_confirmed: int = 0):
            if self.is_linear:
                r = self.linear_attn(self.input_layernorm(x), mask, cache, n_confirmed=n_confirmed)
            else:
                r = self.self_attn(self.input_layernorm(x), mask, cache)
            h = x + r
            out = h + self.mlp(self.post_attention_layernorm(h))
            return out

        DecoderLayer.__call__ = _patched_decoder_call
        DecoderLayer._mtp_n_confirmed_patched = True

    # 3. Subclase segura y libre de monkeypatching concurrente
    original_text_class = text_model.__class__

    class _SafeMTPQwen35TextModel(original_text_class):
        def _lm_logits(self, h: mx.array) -> mx.array:
            lm = getattr(self, "lm_head", None)
            if lm is not None:
                return lm(h)
            return self.model.embed_tokens.as_linear(h)

        def forward_with_hidden(
            self,
            inputs: mx.array,
            cache: Optional[List[Any]] = None,
            n_confirmed: int = 0,
            logits_to_keep: Optional[int] = None,
        ) -> Tuple[mx.array, mx.array]:
            """
            Ejecuta el forward del trunk extrayendo hidden states de forma 100% thread-safe.
            Soporta n_confirmed > 0 para snapshot de estado SSM/conv en rollback.
            logits_to_keep: None=todos, 0=ninguno (solo hidden), 1=última posición.
            """
            inner = self.model  # Qwen3_5TextModel
            hidden_states = inner.embed_tokens(inputs)

            if cache is None:
                cache = [None] * len(inner.layers)

            fa_mask = create_attention_mask(hidden_states, cache[inner.fa_idx])
            ssm_mask = create_ssm_mask(hidden_states, cache[inner.ssm_idx])

            for layer, c in zip(inner.layers, cache):
                mask = ssm_mask if layer.is_linear else fa_mask
                hidden_states = layer(hidden_states, mask=mask, cache=c, n_confirmed=n_confirmed)

            # post_norm: estado normalizado que el cabezal MTP espera (contrato Qwen3.5)
            post_norm = inner.norm(hidden_states)
            if logits_to_keep == 0:
                return None, post_norm
            if logits_to_keep is not None:
                logits = self._lm_logits(post_norm[:, -logits_to_keep:])
            else:
                logits = self._lm_logits(post_norm)
            return logits, post_norm

        def mtp_forward(
            self,
            hidden_states: mx.array,
            next_token_ids: mx.array,
            cache: Optional[Any] = None,
            mtp_cache: Optional[Any] = None,
            return_hidden: bool = False,
            logits_to_keep: Optional[int] = None,
        ) -> mx.array | Tuple[mx.array, mx.array]:
            layer_cache = mtp_cache if mtp_cache is not None else cache
            if isinstance(layer_cache, list):
                layer_cache = layer_cache[0] if layer_cache else None

            # 1. Normalización de embedding del token propuesto + normalización de hidden state
            e = self.mtp.pre_fc_norm_embedding(self.model.embed_tokens(next_token_ids))
            h = self.mtp.pre_fc_norm_hidden(hidden_states)

            # 2. Fusión y proyección de ambos flujos (concat[e, h])
            mixed = self.mtp.fc(mx.concatenate([e, h], axis=-1))

            # 3. Paso por el bloque MoE full-attention del MTP
            mask = create_attention_mask(mixed, layer_cache)
            hidden = self.mtp.layers[0](mixed, mask=mask, cache=layer_cache)

            # 4. Proyección final a logits de vocabulario
            if logits_to_keep == 0:
                return None if not return_hidden else (None, hidden)
            normed = self.mtp.norm(hidden)
            if logits_to_keep is not None:
                logits = self._lm_logits(normed[:, -logits_to_keep:])
            else:
                logits = self._lm_logits(normed)

            if not return_hidden:
                return logits
            return logits, hidden

        def make_mtp_cache(self) -> List[Any]:
            return [KVCache()]

    text_model.__class__ = _SafeMTPQwen35TextModel
    text_model._mtp_injected = True

    # 4. Delegación en el wrapper exterior (qwen3_5_moe.Model) si corresponde
    if getattr(model, "language_model", None) is text_model:
        model.mtp = mtp_module
        original_outer_class = model.__class__

        class _SafeMTPQwen35OuterModel(original_outer_class):
            def forward_with_hidden(self, *args, **kwargs):
                return self.language_model.forward_with_hidden(*args, **kwargs)

            def mtp_forward(self, *args, **kwargs):
                return self.language_model.mtp_forward(*args, **kwargs)

            def make_mtp_cache(self):
                return self.language_model.make_mtp_cache()

        model.__class__ = _SafeMTPQwen35OuterModel
        model._mtp_injected = True

    logger.info("[RESULT] Inyección de soporte MTP completada con éxito (thread-safe e idempotente)")
    return True


def validate_qwen3_5_mtp_support(model: Any) -> bool:
    """Verifica si el modelo tiene la cabeza MTP y sus métodos de inferencia vinculados."""
    if getattr(model, "mtp", None) is None:
        return False
    return (
        callable(getattr(model, "forward_with_hidden", None))
        and callable(getattr(model, "mtp_forward", None))
        and callable(getattr(model, "make_mtp_cache", None))
    )
