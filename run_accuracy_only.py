"""
run_accuracy_only.py
GLUE 정확도만 빠르게 확인하는 스크립트. PPA 계산 없음.

실행:
    python run_accuracy_only.py
"""
import os
import torch
import warnings
warnings.filterwarnings("ignore")

from glue_eval import run_all_glue, GLUE_TASKS, DEVICE

# ============================================================
# 로컬 SST-2 체크포인트 경로 (없으면 HuggingFace hub 사용)
# ============================================================
def _find(candidates):
    return next((p for p in candidates if os.path.exists(p)), None)

SST2_CKPT = _find([
    r"C:\Users\kimsanghyuk\Documents\sanghyuk\W4A8_BERT_best_acc0.9174.pt",
    "/content/W4A8_BERT_best_acc0.9174.pt",
    "/drive/MyDrive/W4A8_BERT_best_acc0.9174.pt",
])

# W4A8 QAT checkpoints (train_glue_w4a8.py로 학습한 결과)
LOCAL_CKPTS = {}
for task, fname in [('mrpc', 'W4A8_MRPC_best.pt'), ('mnli', 'W4A8_MNLI_best.pt')]:
    p = _find([f"/content/PPA/{fname}", f"/content/{fname}",
               f"/drive/MyDrive/{fname}",
               os.path.join(os.path.dirname(__file__), fname)])
    if p:
        LOCAL_CKPTS[task] = p

if __name__ == "__main__":
    print(f"\nDevice : {DEVICE}")
    print("Mode   : Inference only (no training, no PPA)")
    print("Quant  : W4A8 + RRAM variation (Q/K), bias=None")
    if LOCAL_CKPTS:
        print(f"W4A8 ckpts loaded: {list(LOCAL_CKPTS.keys())}")
    print("=" * 60)

    results = run_all_glue(local_sst2_ckpt=SST2_CKPT, local_ckpts=LOCAL_CKPTS)

    # ============================================================
    # 결과 테이블
    # ============================================================
    print("\n" + "=" * 60)
    print(f"{'Task':<8} {'Accuracy':>10}  {'n_val':>7}")
    print("-" * 60)
    for task in GLUE_TASKS:
        r = results[task]
        acc_str = f"{r['accuracy']*100:.2f}%" if r['accuracy'] is not None else "FAILED"
        print(f"{task.upper():<8} {acc_str:>10}  {r['num_samples']:>7}")
    print("=" * 60)
