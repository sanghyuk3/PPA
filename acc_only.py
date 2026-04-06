"""
acc_only.py
GLUE accuracy only (SST-2, MRPC, MNLI) — PPA 계산 없음.
W4A8 QAT checkpoint 사용. RRAM variation 적용 가능.

실행:
    python acc_only.py
"""
import os
import warnings
warnings.filterwarnings("ignore")

from glue_eval import run_all_glue, GLUE_TASKS, DEVICE

# ============================================================
# Checkpoint 경로
# ============================================================
def _find(candidates):
    return next((p for p in candidates if os.path.exists(p)), None)

SST2_CKPT = _find([
    r"C:\Users\kimsanghyuk\Documents\sanghyuk\W4A8_BERT_best_acc0.9174.pt",
    "/content/PPA/W4A8_BERT_best_acc0.9174.pt",
    "/content/W4A8_BERT_best_acc0.9174.pt",
    "/drive/MyDrive/W4A8_BERT_best_acc0.9174.pt",
])

LOCAL_CKPTS = {}
for task, fname in [('mrpc', 'W4A8_MRPC_best.pt'), ('mnli', 'W4A8_MNLI_best.pt')]:
    p = _find([
        os.path.join(os.path.dirname(os.path.abspath(__file__)), fname),
        f"/content/PPA/{fname}",
        f"/content/{fname}",
        f"/drive/MyDrive/{fname}",
    ])
    if p:
        LOCAL_CKPTS[task] = p

# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    import config
    print(f"\nDevice   : {DEVICE}")
    print(f"Quant    : W4A8 + RRAM variation (Q/K)")
    print(f"Variation: STD_Q={config.VARIATION_STD_Q}, STD_K={config.VARIATION_STD_K}")
    print(f"Tasks    : SST-2, MRPC, MNLI")
    if SST2_CKPT:
        print(f"SST2 ckpt: {SST2_CKPT}")
    else:
        print("SST2 ckpt: NOT FOUND — will fail")
    for t, p in LOCAL_CKPTS.items():
        print(f"{t.upper()} ckpt : {p}")
    print("=" * 60)

    results = run_all_glue(local_sst2_ckpt=SST2_CKPT, local_ckpts=LOCAL_CKPTS)

    print("\n" + "=" * 60)
    print(f"{'Task':<8} {'Accuracy':>10}  {'n_val':>7}  {'Source'}")
    print("-" * 60)
    for task in GLUE_TASKS:
        r = results[task]
        acc_str = f"{r['accuracy']*100:.2f}%" if r['accuracy'] is not None else "FAILED"
        if task == 'sst2':
            src = 'W4A8 QAT ckpt' if SST2_CKPT else 'MISSING'
        elif task in LOCAL_CKPTS:
            src = 'W4A8 QAT ckpt'
        else:
            src = 'textattack FP32+PTQ'
        print(f"{task.upper():<8} {acc_str:>10}  {r['num_samples']:>7}  {src}")
    print("=" * 60)
