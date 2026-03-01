import torch
import torch.nn.functional as F
import torch.nn as nn

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
# QuantLinear with Statistics
# ============================================================
class QuantLinearWithStats(nn.Module):
    def __init__(self, old_layer, layer_name="", th_value=0.2, clamp_value=0.6):
        super().__init__()
        self.in_features = old_layer.in_features
        self.out_features = old_layer.out_features
        self.th_value = th_value
        self.clamp_value = clamp_value
        
        self.weight = nn.Parameter(old_layer.weight.data.clone())
        self.bias = None
    
    def forward(self, x):
        x_q = ternary_quantize_pm05(x, 0.33, 0.5)
        w_q = ternary_quantize_pm05(self.weight, self.th_value, self.clamp_value)
        
        output = F.linear(x_q, w_q, self.bias)
        return output

# ============================================================
# Apply Quantization
# ============================================================
def apply_quantlinear_with_stats(model, th_value=0.2, clamp_value=0.6):
    quant_layers = []
    
    for group in model.albert.encoder.albert_layer_groups:
        for albert_layer in group.albert_layers:
            attn = albert_layer.attention
            for name in ["query", "key", "value"]:
                old_linear = getattr(attn, name)
                ql = QuantLinearWithStats(
                    old_linear, layer_name=name, th_value=th_value,
                    clamp_value=clamp_value,
                )
                setattr(attn, name, ql)
                quant_layers.append(ql)
    return quant_layers