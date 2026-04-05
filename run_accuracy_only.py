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
SST2_CKPT = r"C:\Users\kimsanghyuk\Documents\sanghyuk\W4A8_BERT_best_acc0.9174.pt"

if __name__ == "__main__":
    print(f"\nDevice : {DEVICE}")
    print("Mode   : Inference only (no training, no PPA)")
    print("Quant  : W4A8 + RRAM variation (Q/K), bias=None")
    print("=" * 60)

    results = run_all_glue(local_sst2_ckpt=SST2_CKPT)

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
