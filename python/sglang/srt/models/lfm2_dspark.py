"""LFM2-family DSpark draft model.

Registered as its own architecture (rather than reusing Qwen3DSparkModel) so the
LFM2 draft can evolve independently of other DSpark drafts.

Beyond the attention-only Qwen3-style GQA backbone (5L/3L/2L exports), this
supports the LFM2 draft ablations:

- ``config.layer_types`` entries of ``"conv"``: the layer's attention op is
  replaced by an LFM2 gated ShortConv (in_proj -> B,C,x; Bx = B*x; depthwise
  conv1d of width ``config.conv_L_cache``; out = out_proj(C * conv(Bx))). Conv
  layers read no target-context KV and apply no RoPE.
- ``config.conv_causal`` (default true): causal conv (pad left k-1) vs
  symmetric conv (pad k//2 both sides, odd k only).
- ``config.canon_set`` (e.g. "A", "AC") + ``config.canon_kernel``: residual
  depthwise causal conv taps on attention layers — tap A on the normed input
  before QKV, tap C on the normed input of the MLP.

Every conv here mixes tokens along the sequence dim, and the DSpark worker
packs each request's draft block contiguously as (num_blocks * block_len)
tokens — so all convs reshape to (num_blocks, block_len, hidden) first and can
never leak across block boundaries.
"""

from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.models.dflash import DFlashDecoderLayer, DFlashMLP
from sglang.srt.models.dspark import DSparkDraftModel
from sglang.srt.runtime_context import get_parallel


def _split_into_blocks(
    hidden_states: torch.Tensor, num_blocks: int
) -> torch.Tensor:
    """(total_tokens, hidden) -> (num_blocks, hidden, block_len) for conv1d."""
    total_tokens, hidden_size = hidden_states.shape
    if num_blocks <= 0 or total_tokens % num_blocks != 0:
        raise ValueError(
            "LFM2 DSpark conv layers need a uniform block layout: "
            f"total_tokens={total_tokens} is not divisible by "
            f"num_blocks={num_blocks}. Ragged draft layouts are unsupported."
        )
    block_len = total_tokens // num_blocks
    return hidden_states.view(num_blocks, block_len, hidden_size).transpose(1, 2)


def _blockwise_depthwise_conv(
    hidden_states: torch.Tensor,
    *,
    num_blocks: int,
    conv: nn.Conv1d,
    causal: bool,
) -> torch.Tensor:
    """Depthwise conv within each draft block; returns (total_tokens, hidden)."""
    x = _split_into_blocks(hidden_states, num_blocks)
    kernel_size = int(conv.kernel_size[0])
    if causal:
        x = F.pad(x, (kernel_size - 1, 0))
    else:
        half = kernel_size // 2
        x = F.pad(x, (half, half))
    out = conv(x)
    return out.transpose(1, 2).reshape(hidden_states.shape)


def _require_no_tp(what: str) -> None:
    tp_size = int(get_parallel().tp_size)
    if tp_size != 1:
        raise NotImplementedError(
            f"{what} does not support tensor parallelism yet (tp_size={tp_size})."
        )


class Lfm2DSparkShortConv(nn.Module):
    """LFM2 gated ShortConv for draft blocks (weights: in_proj/conv/out_proj)."""

    def __init__(self, *, config) -> None:
        super().__init__()
        _require_no_tp("Lfm2DSparkShortConv")
        hidden_size = int(config.hidden_size)
        kernel_size = int(config.conv_L_cache)
        self.causal = bool(getattr(config, "conv_causal", True))
        if not self.causal and kernel_size % 2 == 0:
            raise ValueError(
                f"Symmetric (non-causal) draft conv requires an odd kernel, "
                f"got conv_L_cache={kernel_size}."
            )
        self.in_proj = nn.Linear(hidden_size, 3 * hidden_size, bias=False)
        self.conv = nn.Conv1d(
            hidden_size, hidden_size, kernel_size, groups=hidden_size, bias=False
        )
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor, num_blocks: int) -> torch.Tensor:
        b_gate, c_gate, x = self.in_proj(hidden_states).chunk(3, dim=-1)
        conv_out = _blockwise_depthwise_conv(
            b_gate * x, num_blocks=num_blocks, conv=self.conv, causal=self.causal
        )
        return self.out_proj(c_gate * conv_out)


class Lfm2DSparkCanon(nn.Module):
    """Residual depthwise causal conv tap (weights: conv)."""

    def __init__(self, *, config) -> None:
        super().__init__()
        _require_no_tp("Lfm2DSparkCanon")
        hidden_size = int(config.hidden_size)
        kernel_size = int(config.canon_kernel)
        self.conv = nn.Conv1d(
            hidden_size, hidden_size, kernel_size, groups=hidden_size, bias=False
        )

    def forward(self, hidden_states: torch.Tensor, num_blocks: int) -> torch.Tensor:
        return hidden_states + _blockwise_depthwise_conv(
            hidden_states, num_blocks=num_blocks, conv=self.conv, causal=True
        )


class Lfm2DSparkConvLayer(nn.Module):
    """Draft layer with the attention op replaced by a gated ShortConv.

    Mirrors DFlashDecoderLayer's (hidden, residual) fused-RMSNorm contract but
    has no self_attn: it reads no context KV and applies no RoPE.
    """

    def __init__(self, *, config, layer_id: int, quant_config=None) -> None:
        super().__init__()
        del layer_id
        hidden_size = int(config.hidden_size)
        rms_norm_eps = float(getattr(config, "rms_norm_eps", 1e-6))
        self.input_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.conv = Lfm2DSparkShortConv(config=config)
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.mlp = DFlashMLP(config=config, quant_config=quant_config)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        del positions
        if hidden_states.numel() == 0:
            if residual is None:
                residual = hidden_states
            return hidden_states, residual

        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        conv_out = self.conv(hidden_states, int(forward_batch.batch_size))
        hidden_states, residual = self.post_attention_layernorm(conv_out, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class Lfm2DSparkCanonDecoderLayer(DFlashDecoderLayer):
    """Attention draft layer with Canon conv taps (weights: canonA/canonC)."""

    def __init__(self, *, config, layer_id: int, quant_config=None) -> None:
        super().__init__(config=config, layer_id=layer_id, quant_config=quant_config)
        canon_set = str(config.canon_set)
        self.canonA = (
            Lfm2DSparkCanon(config=config) if "A" in canon_set else None
        )
        self.canonC = (
            Lfm2DSparkCanon(config=config) if "C" in canon_set else None
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if hidden_states.numel() == 0:
            if residual is None:
                residual = hidden_states
            return hidden_states, residual

        num_blocks = int(forward_batch.batch_size)
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        if self.canonA is not None:
            hidden_states = self.canonA(hidden_states, num_blocks)

        attn_out = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
        )
        hidden_states, residual = self.post_attention_layernorm(attn_out, residual)
        if self.canonC is not None:
            hidden_states = self.canonC(hidden_states, num_blocks)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


def _build_lfm2_dspark_layer(*, config, layer_id: int, quant_config=None):
    layer_types = getattr(config, "layer_types", None)
    layer_type = (
        layer_types[layer_id] if layer_types is not None else "full_attention"
    )
    if layer_type == "conv":
        return Lfm2DSparkConvLayer(
            config=config, layer_id=layer_id, quant_config=quant_config
        )
    canon_set = str(getattr(config, "canon_set", "") or "")
    if canon_set:
        return Lfm2DSparkCanonDecoderLayer(
            config=config, layer_id=layer_id, quant_config=quant_config
        )
    return DFlashDecoderLayer(
        config=config, layer_id=layer_id, quant_config=quant_config
    )


class Lfm2DSparkDraftModel(DSparkDraftModel):

    decoder_layer_cls = staticmethod(_build_lfm2_dspark_layer)

    def kv_context_layers(self) -> list:
        return [
            layer for layer in self.layers if isinstance(layer, DFlashDecoderLayer)
        ]


EntryClass = [Lfm2DSparkDraftModel]
