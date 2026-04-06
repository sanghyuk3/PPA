"""
scaling_analysis.py
문장 수(N) 증가에 따른 PPA 변화 분석

ISAAC   : W_Q/W_K 1회 write (amortized) + N × inference  → energy/sample 감소
Conv RRAM: 문장마다 K^T write            + N × inference  → energy/sample 일정 (높음)
GPU      : N × inference (no write)                       → energy/sample 일정

실행:
    python scaling_analysis.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from param import compute_rram_ppa_for_model
from conventional_rram import (calculate_conventional_rram_ppa_for_model,
                                CONV_V_WRITE_SET, CONV_I_COMPLIANCE, CONV_T_WRITE_SET,
                                CONV_V_WRITE_RESET, CONV_I_RESET, CONV_T_WRITE_RESET)
from GPU.GPU import calculate_gpu_ppa_for_model
from Cell import calculate_rram_write_energy


# ============================================================
# ISAAC 1회 W_Q / W_K write 에너지 (배포 시 1회)
# ============================================================
def isaac_write_once_energy(layers, d_model):
    """W_Q + W_K 전체 weight를 한 번 프로그래밍하는 에너지"""
    E_set   = calculate_rram_write_energy(CONV_V_WRITE_SET,   CONV_I_COMPLIANCE, CONV_T_WRITE_SET)
    E_reset = calculate_rram_write_energy(CONV_V_WRITE_RESET, CONV_I_RESET,      CONV_T_WRITE_RESET)
    E_per_cell = (E_set + E_reset) / 2.0
    total_cells = 2 * d_model * d_model * layers   # W_Q + W_K
    return total_cells * E_per_cell


def isaac_write_once_latency(layers, d_model):
    """W_Q + W_K 전체 write latency (row-parallel)"""
    T_per_row   = CONV_T_WRITE_SET + CONV_T_WRITE_RESET
    total_rows  = 2 * d_model * layers             # W_Q + W_K 각 d_model rows
    return total_rows * T_per_row


# ============================================================
# 모델 스펙
# ============================================================
MODELS = {
    'BERT-base':   {'layers': 12, 'd_model': 768,  'seq_len': 128},
    'GPT-2 Small': {'layers': 12, 'd_model': 768,  'seq_len': 1024},
    'LLaMA-7B':    {'layers': 32, 'd_model': 4096, 'seq_len': 2048},
}

GPU_NAME = 'H100'

N_LIST = [1, 10, 50, 100, 500, 1000, 5000, 10000]


# ============================================================
# 분석
# ============================================================
def run_scaling():
    for model_name, cfg in MODELS.items():
        layers   = cfg['layers']
        d_model  = cfg['d_model']
        seq_len  = cfg['seq_len']

        E_isaac_write_once = isaac_write_once_energy(layers, d_model)
        T_isaac_write_once = isaac_write_once_latency(layers, d_model)

        W = 18
        print("\n" + "="*100)
        print(f"Scaling Analysis: {model_name}  (layers={layers}, d_model={d_model}, seq_len={seq_len})")
        print(f"  ISAAC  one-time write : {E_isaac_write_once*1e3:.2f} mJ  /  {T_isaac_write_once*1e3:.2f} ms")
        print("="*100)
        print(f"{'N':>6}  "
              f"{'ISAAC E/sent(mJ)':>{W}} {'Conv E/sent(mJ)':>{W}} {'GPU E/sent(mJ)':>{W}}  "
              f"{'Conv/ISAAC':>12} {'GPU/ISAAC':>10}")
        print("-"*100)

        for N in N_LIST:
            isaac_r = compute_rram_ppa_for_model(layers, d_model, seq_len, N)
            conv_r  = calculate_conventional_rram_ppa_for_model(layers, d_model, seq_len, N)
            gpu_r   = calculate_gpu_ppa_for_model(layers, d_model, seq_len, N)[GPU_NAME]

            # energy per sentence
            isaac_e = (E_isaac_write_once + isaac_r['energy']) / N   # amortized write + inference
            conv_e  = conv_r['energy'] / N                            # KT write + inference (per sentence)
            gpu_e   = gpu_r['total_energy'] / N

            conv_ratio = conv_e / isaac_e if isaac_e > 0 else float('inf')
            gpu_ratio  = gpu_e  / isaac_e if isaac_e > 0 else float('inf')

            print(f"{N:>6}  "
                  f"{isaac_e*1e3:>{W}.4f} {conv_e*1e3:>{W}.4f} {gpu_e*1e3:>{W}.4f}  "
                  f"{conv_ratio:>11.2f}x {gpu_ratio:>9.2f}x")

        print("="*100)
        print("  ISAAC : 1회 W_Q/W_K write amortized over N sentences (energy/sent decreases with N)")
        print("  Conv  : K^T write every sentence (energy/sent stays constant)")
        print("  GPU   : no write cost (energy/sent stays constant)")

    # ============================================================
    # 추가: BERT-base 기준 latency/sample 비교
    # ============================================================
    cfg = MODELS['BERT-base']
    layers, d_model, seq_len = cfg['layers'], cfg['d_model'], cfg['seq_len']
    E_iw = isaac_write_once_energy(layers, d_model)
    T_iw = isaac_write_once_latency(layers, d_model)

    print("\n" + "="*100)
    print(f"Latency per sentence — BERT-base")
    print("="*100)
    print(f"{'N':>6}  {'ISAAC lat/sent(ms)':>20} {'Conv lat/sent(ms)':>20} {'GPU lat/sent(ms)':>20}")
    print("-"*100)
    for N in N_LIST:
        isaac_r = compute_rram_ppa_for_model(layers, d_model, seq_len, N)
        conv_r  = calculate_conventional_rram_ppa_for_model(layers, d_model, seq_len, N)
        gpu_r   = calculate_gpu_ppa_for_model(layers, d_model, seq_len, N)[GPU_NAME]

        isaac_t = (T_iw + isaac_r['runtime']) / N * 1e3
        conv_t  = conv_r['runtime']             / N * 1e3
        gpu_t   = gpu_r['total_runtime']        / N * 1e3

        print(f"{N:>6}  {isaac_t:>20.4f} {conv_t:>20.4f} {gpu_t:>20.4f}")
    print("="*100)


if __name__ == "__main__":
    run_scaling()
