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
from Inference import apply_quantlinear_with_stats
from param import ISAAC_RRAM_PPA
from evaluation import evaluate_with_ppa
from GPU.GPU import calculate_gpu_qkt_operation

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
    tokenizer = AlbertTokenizer.from_pretrained("albert-base-v2")
    model = AlbertForSequenceClassification.from_pretrained("albert-base-v2", num_labels=2)
    
    # Load checkpoint
    best_model_path = r"C:\Users\kimsanghyuk\Documents\sanghyuk\best_model_case2.pt"
    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path, map_location="cpu")
        if isinstance(checkpoint, dict):
            state_dict = checkpoint.get('model_state_dict', checkpoint.get('state_dict', checkpoint))
        else:
            state_dict = checkpoint
        model.load_state_dict(state_dict, strict=False)
    
    # Apply quantization
    quant_layers = apply_quantlinear_with_stats(model)
    model.eval()
    
    # ============================================================
    # 2. Dataset Setup
    # ============================================================
    print("\n📊 Loading SST-2...")
    dataset = load_dataset("glue", "sst2")
    
    def encode(batch):
        return tokenizer(batch["sentence"], truncation=True, 
                        padding="max_length", max_length=config.RRAM_SENT_LEN)
    
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
    # 6. Comparison Table
    # ============================================================
    gpu_results = calculate_gpu_qkt_operation()
    print("\n" + "="*150)
    print("PPA Comparison Table")
    print("="*150)
    print(f"{'Metric':<25} {'RRAM':<20} {'V100':<20} {'A100':<20} {'H100':<20} {'RTX3090':<20} {'RTX4090':<20} ")
    print("-"*150)
    
    print(f"{'Runtime (ms)':<25} "
          f"{rram_results['runtime']*1e3:<20.3f} "
          f"{gpu_results['V100']['total_runtime']*1e3:<20.3f} "
          f"{gpu_results['A100']['total_runtime']*1e3:<20.3f} "
          f"{gpu_results['H100']['total_runtime']*1e3:<20.3f} "
          f"{gpu_results['RTX_3090']['total_runtime']*1e3:<20.3f} "
          f"{gpu_results['RTX_4090']['total_runtime']*1e3:<20.3f} "
          )
    
    print(f"{'Energy (mJ)':<25} "
          f"{rram_results['energy']*1e3:<20.6f} "
          f"{gpu_results['V100']['total_energy']*1e3:<20.3f} "
          f"{gpu_results['A100']['total_energy']*1e3:<20.3f} "
          f"{gpu_results['H100']['total_energy']*1e3:<20.3f}"
          f"{gpu_results['RTX_3090']['total_energy']*1e3:<20.3f} "
          f"{gpu_results['RTX_4090']['total_energy']*1e3:<20.3f} "
          )
    
    # print(f"{'Power (W)':<25} "
    #       f"{rram_results['power']:<20.6f} "
    #       f"{gpu_results['V100']['power']:<20.6f} "
    #       f"{gpu_results['A100']['power']:<20.6f} "
    #       f"{gpu_results['H100']['power']:<20.6f}"
    #       f"{gpu_results['RTX_3090']['power']:<20.6f} "
    #       f"{gpu_results['RTX_4090']['power']:<20.6f} "
    #       )
    
    # print(f"{'TOPS/W':<25} "
    #       f"{rram_results['TOPS_per_W']:<20.4f} "
    #       f"{gpu_results['V100']['TOPS_per_W']:<20.4f} "
    #       f"{gpu_results['A100']['TOPS_per_W']:<20.4f} "
    #       f"{gpu_results['H100']['TOPS_per_W']:<20.4f}"
    #       f"{gpu_results['RTX_3090']['TOPS_per_W']:<20.4f} "
    #       f"{gpu_results['RTX_4090']['TOPS_per_W']:<20.4f} "
    #       )
    
    print(f"{'Area (mm2)':<25} "
          f"{rram_results['area']:<20.4f} "
          f"{gpu_results['V100']['area']:<20.4f} "
          f"{gpu_results['A100']['area']:<20.1f} "
          f"{gpu_results['H100']['area']:<20.1f}"
          f"{gpu_results['RTX_3090']['area']:<20.1f} "
          f"{gpu_results['RTX_4090']['area']:<20.1f} "
          )
    
    print("="*150 + "\n")