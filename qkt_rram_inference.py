"""
qkt_rram_inference.py
SST-2 첫 번째 문장에 대해 RRAM QK^T variation을 SW로 시뮬레이션하여
긍정/부정 판별.

측정값 (24×24 RRAM, layer0):
  QK^T variation: mean=0.0206, std=0.828  (per element)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from transformers import BertTokenizer, BertForSequenceClassification
from transformers.models.bert.modeling_bert import BertSelfAttention
import os, sys, math

# ============================================================
# W4A8 quantization (학습과 동일)
# ============================================================
def quantize_weight_4bit(w):
    max_abs = w.abs().max().clamp(min=1e-8)
    scale = max_abs / 7.0
    return (w / scale).round().clamp(-7, 7) * scale

def quantize_activation_8bit(x):
    max_abs = x.abs().max().clamp(min=1e-8)
    scale = max_abs / 127.0
    return (x / scale).round().clamp(-127, 127) * scale

# ============================================================
# QK^T RRAM variation이 적용된 커스텀 Attention
# ============================================================
class RRAMBertSelfAttention(nn.Module):
    """BertSelfAttention을 대체. QK^T 결과에 RRAM variation noise 추가."""

    def __init__(self, orig_attn: BertSelfAttention,
                 qkt_mean: float, qkt_std: float, apply_noise: bool = True):
        super().__init__()
        self.num_attention_heads = orig_attn.num_attention_heads
        self.attention_head_size = orig_attn.attention_head_size
        self.all_head_size       = orig_attn.all_head_size

        # W4A8 quantized Q/K/V (bias=None, 학습 조건과 동일)
        self.query = nn.Linear(orig_attn.query.in_features,
                               orig_attn.query.out_features, bias=False)
        self.key   = nn.Linear(orig_attn.key.in_features,
                               orig_attn.key.out_features, bias=False)
        self.value = nn.Linear(orig_attn.value.in_features,
                               orig_attn.value.out_features, bias=False)

        self.query.weight.data.copy_(orig_attn.query.weight.data)
        self.key.weight.data.copy_(orig_attn.key.weight.data)
        self.value.weight.data.copy_(orig_attn.value.weight.data)

        self.dropout    = orig_attn.dropout
        self.qkt_mean   = qkt_mean
        self.qkt_std    = qkt_std
        self.apply_noise = apply_noise

    def transpose_for_scores(self, x):
        B, S, _ = x.size()
        x = x.view(B, S, self.num_attention_heads, self.attention_head_size)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask=None, *args, **kwargs):
        # W4A8 quantization
        h_q = quantize_activation_8bit(hidden_states)
        h_k = quantize_activation_8bit(hidden_states)

        Q = self.transpose_for_scores(
            F.linear(h_q, quantize_weight_4bit(self.query.weight)))
        K = self.transpose_for_scores(
            F.linear(h_k, quantize_weight_4bit(self.key.weight)))
        V = self.transpose_for_scores(
            F.linear(hidden_states, quantize_weight_4bit(self.value.weight)))

        # QK^T
        qkt = torch.matmul(Q, K.transpose(-1, -2))
        qkt = qkt / math.sqrt(self.attention_head_size)

        # RRAM variation noise 적용 (per element i.i.d.)
        if self.apply_noise and self.qkt_std > 0:
            noise = torch.randn_like(qkt) * self.qkt_std + self.qkt_mean
            qkt = qkt + noise

        if attention_mask is not None:
            qkt = qkt + attention_mask

        attn_weights = F.softmax(qkt, dim=-1)
        attn_weights = self.dropout(attn_weights)

        ctx = torch.matmul(attn_weights, V)
        ctx = ctx.permute(0, 2, 1, 3).contiguous()
        ctx = ctx.view(ctx.size(0), ctx.size(1), self.all_head_size)

        return (ctx,)   # BertSelfAttention 출력 형식


def apply_rram_attention(model, qkt_mean, qkt_std, apply_noise=True):
    """모든 레이어의 BertSelfAttention을 RRAMBertSelfAttention으로 교체."""
    for layer in model.bert.encoder.layer:
        orig = layer.attention.self
        layer.attention.self = RRAMBertSelfAttention(
            orig, qkt_mean=qkt_mean, qkt_std=qkt_std, apply_noise=apply_noise)


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 측정된 QK^T variation
    QKT_MEAN = 0.0206
    QKT_STD  = 0.828

    # checkpoint 경로
    _candidates = [
        r"C:\Users\kimsanghyuk\Documents\sanghyuk\W4A8_BERT_best_acc0.9174.pt",
        "/content/PPA/W4A8_BERT_best_acc0.9174.pt",
        "/content/W4A8_BERT_best_acc0.9174.pt",
        "/drive/MyDrive/W4A8_BERT_best_acc0.9174.pt",
    ]
    ckpt_path = next((p for p in _candidates if os.path.exists(p)), None)
    if ckpt_path is None:
        raise FileNotFoundError("W4A8_BERT_best_acc0.9174.pt 를 찾을 수 없습니다.")

    # ============================================================
    # 1. SST-2 첫 번째 문장 확인
    # ============================================================
    print("Loading SST-2 validation set...")
    dataset  = load_dataset("glue", "sst2")["validation"]
    sentence = dataset[0]["sentence"]
    label    = dataset[0]["label"]   # 0=negative, 1=positive
    print(f"\n첫 번째 문장 : \"{sentence}\"")
    print(f"정답 label   : {'POSITIVE' if label == 1 else 'NEGATIVE'} ({label})")

    # ============================================================
    # 2. 모델 로드
    # ============================================================
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=2)

    sd = torch.load(ckpt_path, map_location="cpu")
    if isinstance(sd, dict) and "model_state_dict" in sd:
        sd = sd["model_state_dict"]
    model.load_state_dict(sd, strict=False)

    # RRAM attention (W4A8 + QK^T variation) 적용
    apply_rram_attention(model, qkt_mean=QKT_MEAN, qkt_std=QKT_STD, apply_noise=True)
    model.to(DEVICE).eval()

    # ============================================================
    # 3. Inference
    # ============================================================
    inputs = tokenizer(sentence, return_tensors="pt",
                       padding="max_length", max_length=128,
                       truncation=True)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    torch.manual_seed(42)  # noise 재현성
    with torch.no_grad():
        logits = model(**inputs).logits

    pred  = logits.argmax(dim=-1).item()
    probs = torch.softmax(logits, dim=-1)[0]

    print(f"\n=== RRAM QK^T Inference Result ===")
    print(f"문장     : \"{sentence}\"")
    print(f"예측     : {'POSITIVE' if pred == 1 else 'NEGATIVE'}")
    print(f"정답     : {'POSITIVE' if label == 1 else 'NEGATIVE'}")
    print(f"정확     : {'O' if pred == label else 'X'}")
    print(f"확률     : NEG={probs[0]*100:.1f}%  POS={probs[1]*100:.1f}%")
    print(f"logits   : {logits[0].tolist()}")
    print(f"QK^T variation: mean={QKT_MEAN}, std={QKT_STD}")

    # ============================================================
    # 4. variation=0 비교 (ideal W4A8)
    # ============================================================
    apply_rram_attention(model, qkt_mean=0, qkt_std=0, apply_noise=False)
    with torch.no_grad():
        logits_ideal = model(**inputs).logits
    pred_ideal  = logits_ideal.argmax(dim=-1).item()
    probs_ideal = torch.softmax(logits_ideal, dim=-1)[0]

    print(f"\n=== Ideal W4A8 (no variation) ===")
    print(f"예측     : {'POSITIVE' if pred_ideal == 1 else 'NEGATIVE'}")
    print(f"확률     : NEG={probs_ideal[0]*100:.1f}%  POS={probs_ideal[1]*100:.1f}%")

    # ============================================================
    # 5. SW (FP32) 비교
    # ============================================================
    model_fp32 = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=2)
    sd_fp32 = torch.load(ckpt_path, map_location="cpu")
    if isinstance(sd_fp32, dict) and "model_state_dict" in sd_fp32:
        sd_fp32 = sd_fp32["model_state_dict"]
    model_fp32.load_state_dict(sd_fp32, strict=False)
    model_fp32.to(DEVICE).eval()

    with torch.no_grad():
        logits_fp32 = model_fp32(**inputs).logits
    pred_fp32  = logits_fp32.argmax(dim=-1).item()
    probs_fp32 = torch.softmax(logits_fp32, dim=-1)[0]

    print(f"\n=== SW FP32 (no quantization) ===")
    print(f"예측     : {'POSITIVE' if pred_fp32 == 1 else 'NEGATIVE'}")
    print(f"확률     : NEG={probs_fp32[0]*100:.1f}%  POS={probs_fp32[1]*100:.1f}%")

    # ============================================================
    # 6. 요약 비교표
    # ============================================================
    print(f"\n{'='*55}")
    print(f"{'':20} {'SW (FP32)':>10} {'W4A8':>10} {'W4A8+RRAM':>10}")
    print(f"{'-'*55}")
    print(f"{'예측':20} "
          f"{'POS' if pred_fp32==1 else 'NEG':>10} "
          f"{'POS' if pred_ideal==1 else 'NEG':>10} "
          f"{'POS' if pred==1 else 'NEG':>10}")
    print(f"{'POS 확률(%)':20} "
          f"{probs_fp32[1]*100:>10.1f} "
          f"{probs_ideal[1]*100:>10.1f} "
          f"{probs[1]*100:>10.1f}")
    print(f"{'정답 여부':20} "
          f"{'O' if pred_fp32==label else 'X':>10} "
          f"{'O' if pred_ideal==label else 'X':>10} "
          f"{'O' if pred==label else 'X':>10}")
    print(f"{'='*55}")
    print(f"문장: \"{sentence}\"")
