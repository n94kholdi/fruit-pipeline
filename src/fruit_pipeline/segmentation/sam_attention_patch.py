"""Optional SDPA/FlashAttention backend for the upstream SAM1 attention modules.

``segment_anything`` (the pip dependency, not vendored -- see
``sam_manager.py``) computes attention eagerly: an explicit
``(q @ k.T).softmax(-1) @ v``. This module monkeypatches its two
``Attention.forward`` implementations, at the class level, to instead call
``torch.nn.functional.scaled_dot_product_attention`` (SDPA), which PyTorch
dispatches to a fused kernel -- FlashAttention or memory-efficient attention
on CUDA when the shapes/dtypes qualify, otherwise a fused "math" fallback.
Both replacements are mathematically equivalent to the eager path (same
scale, same softmax, same relative-position bias where present); they should
not change segmentation output beyond ordinary floating-point nondeterminism.

Why not just depend on ``flash-attn`` directly: the mask decoder's
two-way-transformer attention (``transformer.Attention``) has no bias term,
so it is a clean drop-in for the fastest fused kernel. The ViT image
encoder's windowed attention (``image_encoder.Attention``) adds a
content-dependent relative-position bias before the softmax
(``add_decomposed_rel_pos``); FlashAttention's fused kernels do not accept an
arbitrary additive bias, so that path is passed to SDPA via ``attn_mask=``,
which makes PyTorch pick its memory-efficient or math backend instead of the
true FlashAttention kernel for the encoder specifically. It is still a
straightforward, safe swap (one fused op instead of several), just not a
FlashAttention-kernel win for the encoder. This is why the two attention
sites are patched the same way here rather than only wiring in flash-attn
for the decoder alone.

Patching is applied once, at the class level (idempotent), and is gated
behind ``FRUIT_PIPELINE_SAM_SDPA_ATTENTION`` (default off) so it never
changes behavior unless explicitly enabled; any failure to import/patch logs
a warning and leaves the stock eager implementation in place.
"""

from __future__ import annotations

import logging

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

_PATCHED = False


def apply_sdpa_attention() -> bool:
    """Monkeypatch both SAM attention classes to use SDPA. Idempotent.

    Returns True if SDPA attention is active (already patched or patched
    now), False if patching was skipped/failed and the stock eager
    implementation remains in effect.
    """
    global _PATCHED
    if _PATCHED:
        return True
    if not hasattr(F, "scaled_dot_product_attention"):
        logger.warning(
            "FRUIT_PIPELINE_SAM_SDPA_ATTENTION requested but torch %s has no "
            "scaled_dot_product_attention; keeping eager SAM attention",
            torch.__version__,
        )
        return False
    try:
        _patch_image_encoder_attention()
        _patch_transformer_attention()
    except Exception:
        logger.exception(
            "Failed to patch SAM attention to SDPA; keeping eager SAM attention"
        )
        return False
    _PATCHED = True
    logger.info(
        "SAM attention patched to torch.nn.functional.scaled_dot_product_attention "
        "(FlashAttention/memory-efficient kernels where the backend supports them)"
    )
    return True


def _patch_transformer_attention() -> None:
    from segment_anything.modeling.transformer import Attention as TwoWayAttention

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        # Default SDPA scale (1/sqrt(head_dim)) matches the eager path's
        # explicit `/ math.sqrt(c_per_head)`; there is no bias term here.
        out = F.scaled_dot_product_attention(q, k, v)
        out = self._recombine_heads(out)
        return self.out_proj(out)

    TwoWayAttention.forward = forward


def _patch_image_encoder_attention() -> None:
    from segment_anything.modeling.image_encoder import Attention as ViTAttention
    from segment_anything.modeling.image_encoder import add_decomposed_rel_pos

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, _ = x.shape
        qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.reshape(3, B * self.num_heads, H * W, -1).unbind(0)

        attn_mask = None
        if self.use_rel_pos:
            # Same additive bias as the eager path (`add_decomposed_rel_pos`),
            # computed from the unscaled `q` exactly as upstream does, then
            # passed as SDPA's attn_mask instead of being added to a
            # pre-scaled attn matrix by hand.
            bias = q.new_zeros(q.shape[0], H * W, H * W)
            attn_mask = add_decomposed_rel_pos(
                bias, q, self.rel_pos_h, self.rel_pos_w, (H, W), (H, W)
            )

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        x = out.view(B, self.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)
        return self.proj(x)

    ViTAttention.forward = forward
