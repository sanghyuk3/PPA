"""
main.py
Main execution script for RRAM-based transformer inference
"""

import torch
from torch.utils.data import DataLoader
import os
import warnings
import config
from datasets import load_dataset
from transformers import AlbertTokenizer, AlbertForSequenceClassification
from transformers import BertTokenizer, BertForSequenceClassification
from Inference import apply_quantlinear_with_stats
from param import ISAAC_RRAM_PPA, compute_rram_ppa_for_model
from evaluation import evaluate_with_ppa
from GPU.GPU import calculate_gpu_qkt_operation
from conventional_rram import calculate_conventional_rram_ppa, print_conventional_rram_summary
from glue_eval import run_all_glue, GLUE_TASKS, MAX_LEN
from gpt_ppa import run_gpt_ppa_comparison, print_gpt_comparison

# ============================================================
# Environment Variables
# ============================================================
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["PYTHONHASHSEED"] = "218"
warnings.filterwarnings("ignore", message=".*overflowing tokens.*")

# ============================================================
# Main
# ============================================================
if __name__ == "__main__":  
    # ============================================================
    # 1. Model Setup
    # ============================================================
    print("\n📦 Loading model...")
#     tokenizer = AlbertTokenizer.from_pretrained("albert-base-v2")
#     model = AlbertForSequenceClassification.from_pretrained("albert-base-v2", num_labels=2)
    
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
    # Load checkpoint
    _candidates = [
        r"C:\Users\kimsanghyuk\Documents\sanghyuk\W4A8_BERT_best_acc0.9174.pt",
        "/content/W4A8_BERT_best_acc0.9174.pt",
        "/drive/MyDrive/W4A8_BERT_best_acc0.9174.pt",
    ]
    best_model_path = next((p for p in _candidates if os.path.exists(p)), _candidates[0])
    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path, map_location="cpu")
        if isinstance(checkpoint, dict):
            state_dict = checkpoint.get('model_state_dict', checkpoint.get('state_dict', checkpoint))
        else:
            state_dict = checkpoint
        model.load_state_dict(state_dict, strict=False)
    
    # Apply quantization + move to GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    quant_layers = apply_quantlinear_with_stats(model)
    model.to(device)
    model.eval()
    
    # ============================================================
    # 2. Dataset Setup
    # ============================================================
    print("\n📊 Loading SST-2...")
    dataset = load_dataset("glue", "sst2")
    
    def encode(batch):
        return tokenizer(batch["sentence"], truncation=True,
                        padding="max_length", max_length=128)
    
    encoded = dataset["validation"].map(encode, batched=True)
    encoded.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    
    def collate_fn(batch):
        return {
            "input_ids": torch.stack([x["input_ids"] for x in batch]),
            "attention_mask": torch.stack([x["attention_mask"] for x in batch]),
            "labels": torch.tensor([x["label"] for x in batch]),
        }
    
    val_loader = DataLoader(encoded, batch_size=64, shuffle=False, collate_fn=collate_fn)
    
    # ============================================================
    # 3. RRAM PPA Calculation - 실질적으로 연산
    # ============================================================
    print("\n" + "="*80)
    print("RRAM Array PPA Calculation")
    print("="*80)
    
    ppa_calc = ISAAC_RRAM_PPA()
    rram_results = ppa_calc.get_full_results()
    
    # ============================================================
    # 4. Model Evaluation (with accuracy)
    # ============================================================
    print("\n" + "="*80)
    print("Model Evaluation")
    print("="*80)
    
    eval_results = evaluate_with_ppa(model, val_loader, rram_results)
    print(f"Accuracy: {eval_results['accuracy']:.4f} ({eval_results['accuracy']*100:.2f}%)")
    
    # ============================================================
    # 6. Conventional RRAM (KT write) PPA
    # ============================================================
    print("\n" + "="*80)
    print("Conventional RRAM (KT Write) PPA Calculation")
    print("="*80)
    conv_rram_results = calculate_conventional_rram_ppa()
    print_conventional_rram_summary(conv_rram_results)

    # ============================================================
    # 7. Comparison Table
    # ============================================================
    gpu_results = calculate_gpu_qkt_operation()
    W = 22  # column width
    print("\n" + "="*170)
    print("PPA Comparison Table")
    print("="*170)
    print(f"{'Metric':<25} {'ISAAC RRAM':<{W}} {'Conv. RRAM(KT)':<{W}} {'V100':<{W}} {'A100':<{W}} {'H100':<{W}} {'RTX3090':<{W}} {'RTX4090':<{W}}")
    print("-"*170)

    print(f"{'Runtime (ms)':<25} "
          f"{rram_results['runtime']*1e3:<{W}.3f} "
          f"{conv_rram_results['runtime']*1e3:<{W}.3f} "
          f"{gpu_results['V100']['total_runtime']*1e3:<{W}.3f} "
          f"{gpu_results['A100']['total_runtime']*1e3:<{W}.3f} "
          f"{gpu_results['H100']['total_runtime']*1e3:<{W}.3f} "
          f"{gpu_results['RTX_3090']['total_runtime']*1e3:<{W}.3f} "
          f"{gpu_results['RTX_4090']['total_runtime']*1e3:<{W}.3f}"
          )

    print(f"{'  (write ms)':<25} "
          f"{'N/A':<{W}} "
          f"{conv_rram_results['T_write']*1e3:<{W}.3f} "
          f"{'N/A':<{W}} {'N/A':<{W}} {'N/A':<{W}} {'N/A':<{W}} {'N/A':<{W}}"
          )

    print(f"{'Energy (mJ)':<25} "
          f"{rram_results['energy']*1e3:<{W}.4f} "
          f"{conv_rram_results['energy']*1e3:<{W}.4f} "
          f"{gpu_results['V100']['total_energy']*1e3:<{W}.3f} "
          f"{gpu_results['A100']['total_energy']*1e3:<{W}.3f} "
          f"{gpu_results['H100']['total_energy']*1e3:<{W}.3f} "
          f"{gpu_results['RTX_3090']['total_energy']*1e3:<{W}.3f} "
          f"{gpu_results['RTX_4090']['total_energy']*1e3:<{W}.3f}"
          )

    print(f"{'  (write mJ)':<25} "
          f"{'N/A':<{W}} "
          f"{conv_rram_results['E_write']*1e3:<{W}.4f} "
          f"{'N/A':<{W}} {'N/A':<{W}} {'N/A':<{W}} {'N/A':<{W}} {'N/A':<{W}}"
          )

    print(f"{'TOPS':<25} "
          f"{rram_results['TOPS']:<{W}.4f} "
          f"{conv_rram_results['TOPS']:<{W}.4f} "
          f"{gpu_results['V100']['TOPS']:<{W}.4f} "
          f"{gpu_results['A100']['TOPS']:<{W}.4f} "
          f"{gpu_results['H100']['TOPS']:<{W}.4f} "
          f"{gpu_results['RTX_3090']['TOPS']:<{W}.4f} "
          f"{gpu_results['RTX_4090']['TOPS']:<{W}.4f}"
          )

    print(f"{'TOPS/W':<25} "
          f"{rram_results['TOPS_per_W']:<{W}.4f} "
          f"{conv_rram_results['TOPS_per_W']:<{W}.4f} "
          f"{gpu_results['V100']['TOPS_per_W']:<{W}.4f} "
          f"{gpu_results['A100']['TOPS_per_W']:<{W}.4f} "
          f"{gpu_results['H100']['TOPS_per_W']:<{W}.4f} "
          f"{gpu_results['RTX_3090']['TOPS_per_W']:<{W}.4f} "
          f"{gpu_results['RTX_4090']['TOPS_per_W']:<{W}.4f}"
          )

    print(f"{'Area (mm2)':<25} "
          f"{rram_results['area']:<{W}.4f} "
          f"{conv_rram_results['area']:<{W}.4f} "
          f"{gpu_results['V100']['area']:<{W}.4f} "
          f"{gpu_results['A100']['area']:<{W}.1f} "
          f"{gpu_results['H100']['area']:<{W}.1f} "
          f"{gpu_results['RTX_3090']['area']:<{W}.1f} "
          f"{gpu_results['RTX_4090']['area']:<{W}.1f}"
          )

    print(f"{'TOPS/mm2':<25} "
          f"{rram_results['TOPS_per_mm2']:<{W}.4f} "
          f"{conv_rram_results['TOPS_per_mm2']:<{W}.4f} "
          f"{gpu_results['V100']['TOPS_per_mm2']:<{W}.4f} "
          f"{gpu_results['A100']['TOPS_per_mm2']:<{W}.4f} "
          f"{gpu_results['H100']['TOPS_per_mm2']:<{W}.4f} "
          f"{gpu_results['RTX_3090']['TOPS_per_mm2']:<{W}.4f} "
          f"{gpu_results['RTX_4090']['TOPS_per_mm2']:<{W}.4f}"
          )

    print("="*170 + "\n")

    # ============================================================
    # 8. Multi-GLUE Evaluation (BERT + RRAM variation)
    # ============================================================
    print("\n" + "="*80)
    print("Multi-GLUE Evaluation (BERT-base + RRAM variation)")
    print("="*80)
    # W4A8 QAT checkpoints (train_glue_w4a8.py로 학습한 결과)
    _base = os.path.dirname(os.path.abspath(__file__))
    local_ckpts = {}
    for task, fname in [('mrpc', 'W4A8_MRPC_best.pt'), ('mnli', 'W4A8_MNLI_best.pt')]:
        p = os.path.join(_base, fname)
        if os.path.exists(p):
            local_ckpts[task] = p

    glue_results = run_all_glue(local_sst2_ckpt=best_model_path, local_ckpts=local_ckpts)

    # 각 task별 PPA: 동일 BERT 모델, task마다 sample 수만 다름
    glue_ppa = {}
    for task, gres in glue_results.items():
        n = gres['num_samples']
        if n > 0:
            glue_ppa[task] = compute_rram_ppa_for_model(
                layers=12, d_model=768, sent_len=MAX_LEN, num_samples=n
            )

    W2 = 14
    print("\n" + "="*100)
    print(f"GLUE Summary  (BERT-base RRAM, seq_len={MAX_LEN})")
    print("="*100)
    print(f"{'Task':<8} {'n_val':<7} {'Accuracy':<12} "
          f"{'Runtime(ms)':<{W2}} {'Energy(mJ)':<{W2}} "
          f"{'TOPS':<{W2}} {'TOPS/W':<{W2}}")
    print("-"*100)
    for task in GLUE_TASKS:
        g = glue_results[task]
        p = glue_ppa.get(task)
        acc_str = f"{g['accuracy']*100:.2f}%" if g['accuracy'] is not None else "N/A"
        if p:
            print(f"{task.upper():<8} {g['num_samples']:<7} {acc_str:<12} "
                  f"{p['runtime']*1e3:<{W2}.3f} {p['energy']*1e3:<{W2}.4f} "
                  f"{p['TOPS']:<{W2}.4f} {p['TOPS_per_W']:<{W2}.4f}")
        else:
            print(f"{task.upper():<8} {g['num_samples']:<7} {acc_str:<12} "
                  f"{'N/A':<{W2}} {'N/A':<{W2}} {'N/A':<{W2}} {'N/A':<{W2}}")
    print("="*100)

    # ============================================================
    # 9. GPT-2 RRAM PPA vs GPU Comparison
    # ============================================================
    print("\n" + "="*80)
    print("GPT-2 RRAM PPA vs GPU Comparison")
    print("="*80)
    gpt_results = run_gpt_ppa_comparison()
    print_gpt_comparison(gpt_results)