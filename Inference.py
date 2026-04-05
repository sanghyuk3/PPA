"""
Inference.py
W4A8 quantization + RRAM variation noise for BERT/ALBERT
  - W4: symmetric per-channel 4-bit weight quantization
  - A8: symmetric per-tensor 8-bit activation quantization
  - RRAM variation: fixed noise sampled once at init (chip programming error)
"""
import torch
import torch.nn.functional as F
import torch.nn as nn
import config


# ============================================================
# Quantization Utilities
# ============================================================
def quantize_weight_4bit(w):
    """
    Symmetric per-tensor 4-bit quantization. Matches WeightQuantSTE in training.
    Range: [-7, 7], dequantized back to float.
    """
    max_abs = w.abs().max().clamp(min=1e-8)  # per-tensor (not per-channel)
    scale = max_abs / 7.0
    q = (w / scale).round().clamp(-7, 7)
    return q * scale


def quantize_activation_8bit(x):
    """
    Symmetric per-tensor 8-bit activation quantization. Matches ActQuantSTE in training.
    Range: [-127, 127], dequantized back to float.
    """
    max_abs = x.abs().max().clamp(min=1e-8)
    scale = max_abs / 127.0
    q = (x / scale).round().clamp(-127, 127)
    return q * scale


# ============================================================
# QuantLinear: W4A8 + RRAM Variation
# ============================================================
class QuantLinearWithStats(nn.Module):
    def __init__(self, old_layer, layer_name="", force_no_bias=False):
        super().__init__()
        self.in_features  = old_layer.in_features
        self.out_features = old_layer.out_features
        self.layer_name   = layer_name

        self.weight = nn.Parameter(old_layer.weight.data.clone())
        if force_no_bias or old_layer.bias is None:
            self.bias = None
        else:
            self.bias = nn.Parameter(old_layer.bias.data.clone())

        # RRAM 프로그래밍 오차 (칩 제작 시 1회 고정)
        # Q/K path만 RRAM variation 적용, V는 zero noise
        if layer_name == "query":
            noise = (torch.randn(self.weight.shape) * config.VARIATION_STD_Q
                     + config.VARIATION_MEAN_Q)
        elif layer_name == "key":
            noise = (torch.randn(self.weight.shape) * config.VARIATION_STD_K
                     + config.VARIATION_MEAN_K)
        else:
            noise = torch.zeros(self.weight.shape)

        self.register_buffer('rram_variation', noise)

    def forward(self, x):
        x_q   = quantize_activation_8bit(x)              # A8
        w_q   = quantize_weight_4bit(self.weight.data)   # W4
        w_rram = w_q + self.rram_variation                # RRAM variation 적용
        return F.linear(x_q, w_rram, self.bias)


# ============================================================
# Apply Quantization (BERT / ALBERT 공통 지원)
# ============================================================
def apply_quantlinear_with_stats(model, force_no_bias=False):
    quant_layers = []

    if hasattr(model, 'bert'):
        for layer in model.bert.encoder.layer:
            attn = layer.attention.self
            for name in ["query", "key", "value"]:
                old_linear = getattr(attn, name)
                ql = QuantLinearWithStats(old_linear, layer_name=name,
                                         force_no_bias=force_no_bias)
                setattr(attn, name, ql)
                quant_layers.append(ql)

    elif hasattr(model, 'albert'):
        for group in model.albert.encoder.albert_layer_groups:
            for albert_layer in group.albert_layers:
                attn = albert_layer.attention
                for name in ["query", "key", "value"]:
                    old_linear = getattr(attn, name)
                    ql = QuantLinearWithStats(old_linear, layer_name=name,
                                             force_no_bias=force_no_bias)
                    setattr(attn, name, ql)
                    quant_layers.append(ql)

    return quant_layers
