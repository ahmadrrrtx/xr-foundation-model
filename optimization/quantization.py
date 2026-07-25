"""
Weight quantization for XRFM (v0.9.0).

Provides INT8 and INT4 weight-only quantization for model compression.
Supports per-tensor, per-channel, and group-wise quantization.

INT8: 4x compression (32→8 bits), ~2x speedup on CPU with FBGEMM.
INT4: 8x compression (32→4 bits), requires group-wise for accuracy.

All quantizers follow the affine (asymmetric) scheme:
    q = round(x / scale) + zero_point
    x_hat = (q - zero_point) * scale

Conceptual references (not copied):
- Jacob et al. (2018) — Quantization and Training of Neural Networks
- Dettmers et al. (2022) — LLM.int8(): 8-bit Matrix Multiplication
- Frantar et al. (2023) — GPTQ: Accurate Post-Training Quantization
- Lin et al. (2024) — AWQ: Activation-aware Weight Quantization

Implementation is original.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class QuantizedWeight:
    """Container for quantized weight + metadata.

    Attributes:
        data: Packed quantized values (torch.int8 for INT8, packed int8 for INT4).
        scale: Per-group or per-channel scale factors (float32).
        zero_point: Per-group or per-channel zero points (int32 or None for symmetric).
        original_shape: Original weight shape before quantization.
        bits: Bit width (4 or 8).
        group_size: Group size for group-wise quantization (-1 = per-channel).
        symmetric: Whether symmetric quantization was used (zero_point is None).
    """

    data: torch.Tensor
    scale: torch.Tensor
    zero_point: torch.Tensor | None
    original_shape: torch.Size
    bits: int
    group_size: int
    symmetric: bool

    def dequantize(self) -> torch.Tensor:
        """Reconstruct approximate FP32 weight from quantized data."""
        return dequantize_weight(self)


# ---------------------------------------------------------------------------
# Core quantization functions
# ---------------------------------------------------------------------------


def _compute_scale_zp(
    tensor: torch.Tensor,
    qmin: int,
    qmax: int,
    symmetric: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Compute scale and zero_point for quantization.

    Affine:  scale = (rmax - rmin) / (qmax - qmin)
             zp = round(qmin - rmin/scale)

    Symmetric: scale = max(|rmax|, |rmin|) / qmax, zp = 0
    """
    rmin = tensor.min()
    rmax = tensor.max()

    if symmetric:
        absmax = max(abs(rmax.item()), abs(rmin.item()))
        if absmax == 0:
            absmax = 1.0
        scale = torch.tensor(absmax / qmax, dtype=torch.float32)
        return scale, None  # zero_point is implicitly 0

    qrange = qmax - qmin
    if qrange == 0:
        qrange = 1
    scale = (rmax - rmin) / qrange
    if scale.item() == 0:
        scale = torch.tensor(1.0, dtype=torch.float32)

    zp = qmin - torch.round(rmin / scale)
    zp = torch.clamp(zp, qmin, qmax).to(torch.int32)
    return scale, zp


def _quantize_tensor(
    tensor: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor | None,
    qmin: int,
    qmax: int,
) -> torch.Tensor:
    """Quantize a tensor given scale and zero_point."""
    if zero_point is not None:
        q = torch.round(tensor / scale + zero_point.float())
    else:
        q = torch.round(tensor / scale)
    return torch.clamp(q, qmin, qmax)


def quantize_int8_per_tensor(
    weight: torch.Tensor,
    symmetric: bool = False,
) -> QuantizedWeight:
    """Quantize weight to INT8 with a single scale/zp for the entire tensor.

    Args:
        weight: FP32 weight tensor of any shape.
        symmetric: If True, use symmetric quantization (zp=0).

    Returns:
        QuantizedWeight with INT8 data.
    """
    qmin, qmax = -128, 127
    scale, zp = _compute_scale_zp(weight, qmin, qmax, symmetric=symmetric)

    q_weight = _quantize_tensor(weight, scale, zp, qmin, qmax).to(torch.int8)

    return QuantizedWeight(
        data=q_weight,
        scale=scale,
        zero_point=zp,
        original_shape=weight.shape,
        bits=8,
        group_size=-1,
        symmetric=symmetric,
    )


def quantize_int8_per_channel(
    weight: torch.Tensor,
    axis: int = 0,
    symmetric: bool = False,
) -> QuantizedWeight:
    """Quantize weight to INT8 with per-channel (per-row) scales.

    Args:
        weight: FP32 weight tensor (typically [out_features, in_features]).
        axis: Dimension along which to compute per-channel stats.
        symmetric: If True, use symmetric quantization.

    Returns:
        QuantizedWeight with INT8 data and per-channel scales.
    """
    qmin, qmax = -128, 127
    out_dim = weight.shape[axis]

    scales = torch.zeros(out_dim, dtype=torch.float32)
    zps = torch.zeros(out_dim, dtype=torch.int32) if not symmetric else None

    q_weight = torch.zeros_like(weight, dtype=torch.float32)

    for i in range(out_dim):
        slc = weight.select(axis, i)
        s, zp = _compute_scale_zp(slc, qmin, qmax, symmetric=symmetric)
        scales[i] = s
        if zps is not None:
            zps[i] = zp
        q_slice = _quantize_tensor(slc, s, zp, qmin, qmax)
        q_weight.select(axis, i).copy_(q_slice)

    return QuantizedWeight(
        data=q_weight.to(torch.int8),
        scale=scales,
        zero_point=zps,
        original_shape=weight.shape,
        bits=8,
        group_size=-1,
        symmetric=symmetric,
    )


def quantize_int4_groupwise(
    weight: torch.Tensor,
    group_size: int = 128,
    symmetric: bool = True,
) -> QuantizedWeight:
    """Quantize weight to INT4 with group-wise quantization using vectorized PyTorch ops.

    For INT4, group-wise quantization is essential to maintain accuracy
    since 4 bits have very limited representational capacity.

    Each group of `group_size` elements gets its own scale.
    INT4 values are packed: 2 values per int8 byte.

    Args:
        weight: FP32 weight tensor.
        group_size: Number of elements per quantization group.
        symmetric: If True, use symmetric quantization (recommended for INT4).

    Returns:
        QuantizedWeight with packed INT4 data and per-group scales.
    """
    if group_size <= 0:
        raise ValueError(f"group_size must be positive, got {group_size}")

    original_shape = weight.shape
    flat = weight.reshape(-1).float()
    n_elements = flat.numel()

    # Pad to multiple of group_size
    pad = (group_size - (n_elements % group_size)) % group_size
    if pad > 0:
        flat = torch.cat([flat, torch.zeros(pad, device=flat.device)])

    n_groups = flat.numel() // group_size
    qmin, qmax = -8, 7  # INT4 range

    # Vectorized group-wise scale computation
    grouped = flat.reshape(n_groups, group_size)
    absmax = grouped.abs().max(dim=1, keepdim=True).values
    absmax = torch.where(absmax == 0, torch.ones_like(absmax), absmax)
    scales = (absmax / qmax).squeeze(1)

    # Vectorized quantization
    q_grouped = torch.clamp(torch.round(grouped / scales.unsqueeze(1)), qmin, qmax).to(torch.int8)
    q_flat = q_grouped.reshape(-1)

    # Vectorized nibble packing: two INT4 values per int8 byte
    even = q_flat[0::2] & 0xF
    odd = q_flat[1::2] & 0xF
    packed = (even | (odd << 4)).to(torch.int8)

    return QuantizedWeight(
        data=packed,
        scale=scales,
        zero_point=None,  # symmetric
        original_shape=original_shape,
        bits=4,
        group_size=group_size,
        symmetric=True,
    )


# ---------------------------------------------------------------------------
# Dequantization
# ---------------------------------------------------------------------------


def dequantize_weight(qw: QuantizedWeight) -> torch.Tensor:
    """Reconstruct approximate FP32 tensor from quantized weight."""
    if qw.bits == 8:
        if qw.group_size < 0 and qw.scale.numel() == 1:
            # Per-tensor
            flat = qw.data.float()
            if qw.zero_point is not None:
                flat = (flat - qw.zero_point.float()) * qw.scale
            else:
                flat = flat * qw.scale
        else:
            # Per-channel
            flat = qw.data.float()
            for i in range(qw.scale.numel()):
                s = qw.scale[i]
                zp = qw.zero_point[i] if qw.zero_point is not None else 0
                flat[i] = (flat[i] - zp) * s
        return flat.reshape(qw.original_shape)

    elif qw.bits == 4:
        # Vectorized INT4 unpacking
        data_int = qw.data.to(torch.int32)
        low = data_int & 0xF
        high = (data_int >> 4) & 0xF

        # Sign-extend 4-bit signed ints (-8 to 7)
        low = torch.where(low > 7, low - 16, low).float()
        high = torch.where(high > 7, high - 16, high).float()

        # Interleave low and high nibbles
        unpacked = torch.stack([low, high], dim=1).reshape(-1)

        # Vectorized group-wise scaling
        n_groups = qw.scale.numel()
        group_size = qw.group_size
        scaled = (unpacked.reshape(n_groups, group_size) * qw.scale.unsqueeze(1)).reshape(-1)

        n_orig = math.prod(qw.original_shape)
        return scaled[:n_orig].reshape(qw.original_shape)

    raise ValueError(f"Unsupported bit width: {qw.bits}")


# ---------------------------------------------------------------------------
# Model quantization
# ---------------------------------------------------------------------------


def quantize_model_weights(
    model: nn.Module,
    bits: int = 8,
    group_size: int = 128,
    per_channel: bool = True,
    symmetric: bool = True,
) -> tuple[nn.Module, dict[str, QuantizedWeight]]:
    """Quantize all Linear layer weights in a model.

    Args:
        model: PyTorch model with nn.Linear layers.
        bits: 8 for INT8, 4 for INT4.
        group_size: Group size for INT4 group-wise quantization.
        per_channel: Use per-channel quantization for INT8.
        symmetric: Use symmetric quantization.

    Returns:
        (model_with_quantized_weights, quantization_map)
        where quantization_map maps parameter names to QuantizedWeight objects.
    """
    if bits not in (4, 8):
        raise ValueError(f"bits must be 4 or 8, got {bits}")

    q_map: dict[str, QuantizedWeight] = {}

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            weight = module.weight.data.float()

            if bits == 8:
                if per_channel:
                    qw = quantize_int8_per_channel(weight, symmetric=symmetric)
                else:
                    qw = quantize_int8_per_tensor(weight, symmetric=symmetric)
            else:  # bits == 4
                qw = quantize_int4_groupwise(weight, group_size=group_size, symmetric=True)

            q_map[name] = qw

    return model, q_map


def compute_compression_ratio(
    q_map: dict[str, QuantizedWeight],
) -> float:
    """Compute compression ratio: original_size / quantized_size."""
    original_bits = 0
    quantized_bits = 0

    for qw in q_map.values():
        n_params = math.prod(qw.original_shape)
        original_bits += n_params * 32
        quantized_bits += n_params * qw.bits
        # Add scale storage
        quantized_bits += qw.scale.numel() * 32
        if qw.zero_point is not None:
            quantized_bits += qw.zero_point.numel() * 32

    if quantized_bits == 0:
        return 1.0
    return original_bits / quantized_bits
