import torch
import torch.nn.functional as F
import torch.nn as nn
import config

class TernaryQuantizeFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor, threshold, scale):
        q_tensor = torch.zeros_like(tensor)
        q_tensor[tensor > threshold] = scale
        q_tensor[tensor < -threshold] = -scale
        return q_tensor

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clone(), None, None

def ternary_quantize_pm05(tensor, threshold, scale):
    return TernaryQuantizeFn.apply(tensor, threshold, scale)

# ============================================================
# QuantLinear with RRAM Variation
# ============================================================
class QuantLinearWithStats(nn.Module):
    def __init__(self, old_layer, layer_name="", th_value=0.2, clamp_value=0.6):
        super().__init__()
        self.in_features = old_layer.in_features
        self.out_features = old_layer.out_features
        self.th_value = th_value
        self.clamp_value = clamp_value
        self.layer_name = layer_name

        self.weight = nn.Parameter(old_layer.weight.data.clone())
        self.bias = None

        # RRAM 프로그래밍 오차 (tuning error) - 칩 프로그래밍 시 1회 고정, inference마다 동일
        # Q/K path 별로 측정된 통계(24×24 → 768×768 동일 per-cell stats 적용)
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
        x_q = ternary_quantize_pm05(x, 0.33, 0.5)
        w_q = ternary_quantize_pm05(self.weight, self.th_value, self.clamp_value)
        # 프로그래밍 오차 적용: w_effective = w_ternary + Δ
        w_rram = w_q + self.rram_variation
        output = F.linear(x_q, w_rram, self.bias)
        return output

# ============================================================
# Apply Quantization (BERT / ALBERT 공통 지원)
# ============================================================
def apply_quantlinear_with_stats(model, th_value=0.2, clamp_value=0.6):
    quant_layers = []

    if hasattr(model, 'bert'):
        # BERT
        for layer in model.bert.encoder.layer:
            attn = layer.attention.self
            for name in ["query", "key", "value"]:
                old_linear = getattr(attn, name)
                ql = QuantLinearWithStats(
                    old_linear, layer_name=name,
                    th_value=th_value, clamp_value=clamp_value,
                )
                setattr(attn, name, ql)
                quant_layers.append(ql)

    elif hasattr(model, 'albert'):
        # ALBERT
        for group in model.albert.encoder.albert_layer_groups:
            for albert_layer in group.albert_layers:
                attn = albert_layer.attention
                for name in ["query", "key", "value"]:
                    old_linear = getattr(attn, name)
                    ql = QuantLinearWithStats(
                        old_linear, layer_name=name,
                        th_value=th_value, clamp_value=clamp_value,
                    )
                    setattr(attn, name, ql)
                    quant_layers.append(ql)

    return quant_layers
