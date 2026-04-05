"""
check_ckpt.py
checkpoint 안의 key 구조를 확인하는 진단 스크립트.
"""
import torch
from transformers import BertForSequenceClassification

CKPT_PATH = r"C:\Users\kimsanghyuk\Documents\sanghyuk\W4A8_BERT_best_acc0.9174.pt"

ckpt = torch.load(CKPT_PATH, map_location='cpu')

print("=== checkpoint top-level type ===")
print(type(ckpt))

if isinstance(ckpt, dict):
    print("\n=== top-level keys ===")
    for k in ckpt.keys():
        print(f"  {k}")

    # state dict 찾기
    if 'model_state_dict' in ckpt:
        sd = ckpt['model_state_dict']
        label = 'model_state_dict'
    elif 'state_dict' in ckpt:
        sd = ckpt['state_dict']
        label = 'state_dict'
    else:
        sd = ckpt
        label = '(ckpt itself)'
else:
    sd = ckpt
    label = '(raw state dict)'

print(f"\n=== state dict source: {label} ===")
print(f"총 key 수: {len(sd)}")
print("\n처음 15개 key:")
for i, k in enumerate(list(sd.keys())[:15]):
    v = sd[k]
    print(f"  [{i:02d}] {k}  → shape={tuple(v.shape) if hasattr(v, 'shape') else v}")

# 표준 BERT key와 비교
print("\n=== 표준 BERT key vs checkpoint key 매칭 ===")
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
model_keys = set(model.state_dict().keys())
ckpt_keys  = set(sd.keys())

matched   = model_keys & ckpt_keys
only_ckpt = ckpt_keys  - model_keys
only_model = model_keys - ckpt_keys

print(f"  매칭된 key:       {len(matched)}")
print(f"  checkpoint에만:   {len(only_ckpt)}")
print(f"  model에만(없는):  {len(only_model)}")

if only_ckpt:
    print("\ncheckpoint에만 있는 key (처음 10개):")
    for k in list(only_ckpt)[:10]:
        print(f"  {k}")

if only_model:
    print("\nmodel에만 있는 key (처음 10개):")
    for k in list(only_model)[:10]:
        print(f"  {k}")
