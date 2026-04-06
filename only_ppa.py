"""
only_ppa.py
RRAM PPA vs GPU 비교 - 모델별 확장 (accuracy 계산 없음)

실행:
    python only_ppa.py
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from param import compute_rram_ppa_for_model
from GPU.GPU import calculate_gpu_ppa_for_model
from conventional_rram import (calculate_conventional_rram_ppa_for_model,
                                CONV_V_WRITE_SET, CONV_I_COMPLIANCE, CONV_T_WRITE_SET,
                                CONV_V_WRITE_RESET, CONV_I_RESET, CONV_T_WRITE_RESET)
from Cell import calculate_rram_write_energy

# ============================================================
# 모델 스펙 정의
# ============================================================
# (layers, d_model, seq_len, label)
# seq_len: BERT=128, GPT-2=1024, LLaMA=2048
MODELS = {
    # Encoder (BERT)
    'BERT-base':    {'layers': 12, 'd_model': 768,  'seq_len': 128},

    # Decoder (GPT-2)
    'GPT-2 Small':  {'layers': 12, 'd_model': 768,  'seq_len': 1024},
    'GPT-2 Medium': {'layers': 24, 'd_model': 1024, 'seq_len': 1024},
    'GPT-2 Large':  {'layers': 36, 'd_model': 1280, 'seq_len': 1024},
    'GPT-2 XL':     {'layers': 48, 'd_model': 1600, 'seq_len': 1024},

    # Large-scale Decoder (LLaMA)
    'LLaMA-7B':     {'layers': 32, 'd_model': 4096, 'seq_len': 2048},
    'LLaMA-13B':    {'layers': 40, 'd_model': 5120, 'seq_len': 2048},
}

NUM_SAMPLES = 1  # per-inference PPA (1 sample 기준)

GPU_LIST = ['V100', 'A100', 'H100', 'RTX_3090', 'RTX_4090']

SCALING_N_LIST = [1, 10, 50, 100, 500, 1000, 5000, 10000]
SCALING_MODEL  = 'BERT-base'   # scaling 분석 기준 모델
SCALING_GPU    = 'H100'


def _isaac_write_once(layers, d_model):
    """W_Q + W_K 1회 프로그래밍 에너지 / 레이턴시"""
    E_per_cell = (calculate_rram_write_energy(CONV_V_WRITE_SET, CONV_I_COMPLIANCE, CONV_T_WRITE_SET) +
                  calculate_rram_write_energy(CONV_V_WRITE_RESET, CONV_I_RESET, CONV_T_WRITE_RESET)) / 2.0
    cells = 2 * d_model * d_model * layers          # W_Q + W_K
    T_per_row = CONV_T_WRITE_SET + CONV_T_WRITE_RESET
    return cells * E_per_cell, cells / d_model * T_per_row   # (energy_J, latency_s)

# ============================================================
# PPA 계산
# ============================================================
def run():
    rram_results = {}
    conv_results = {}
    gpu_results  = {}

    for name, cfg in MODELS.items():
        rram_results[name] = compute_rram_ppa_for_model(
            layers=cfg['layers'],
            d_model=cfg['d_model'],
            sent_len=cfg['seq_len'],
            num_samples=NUM_SAMPLES
        )
        conv_results[name] = calculate_conventional_rram_ppa_for_model(
            layers=cfg['layers'],
            d_model=cfg['d_model'],
            sent_len=cfg['seq_len'],
            num_samples=NUM_SAMPLES
        )
        gpu_results[name] = calculate_gpu_ppa_for_model(
            layers=cfg['layers'],
            d_model=cfg['d_model'],
            sent_len=cfg['seq_len'],
            num_samples=NUM_SAMPLES
        )

    # ============================================================
    # 출력: RRAM (ISAAC vs Conventional)
    # ============================================================
    W = 14
    print("\n" + "="*130)
    print("RRAM PPA — per inference (1 sample)")
    print("="*130)
    print(f"{'Model':<16} {'Layers':>6} {'d_model':>8} {'seq_len':>8} "
          f"{'':>4} {'Runtime(ms)':>{W}} {'Energy(mJ)':>{W}} "
          f"{'TOPS':>{W}} {'TOPS/W':>{W}} {'TOPS/mm2':>{W}}")
    print("-"*130)
    for name, cfg in MODELS.items():
        r = rram_results[name]
        c = conv_results[name]
        print(f"{name:<16} {cfg['layers']:>6} {cfg['d_model']:>8} {cfg['seq_len']:>8} "
              f"{'ISAAC':>4} "
              f"{r['runtime']*1e3:>{W}.4f} {r['energy']*1e3:>{W}.6f} "
              f"{r['TOPS']:>{W}.4f} {r['TOPS_per_W']:>{W}.2f} {r['TOPS_per_mm2']:>{W}.4f}")
        print(f"{'':16} {'':>6} {'':>8} {'':>8} "
              f"{'Conv':>4} "
              f"{c['runtime']*1e3:>{W}.4f} {c['energy']*1e3:>{W}.6f} "
              f"{c['TOPS']:>{W}.4f} {c['TOPS_per_W']:>{W}.2f} {c['TOPS_per_mm2']:>{W}.4f}")
        print(f"{'':16} {'':>6} {'':>8} {'':>8} "
              f"{'':>4} "
              f"{'  (write ms)':>{W}} {c['T_write']*1e3:>{W}.4f}"
              f"{'  (write mJ)':>{W}} {c['E_write']*1e3:>{W}.6f}")
        print()
    print("="*130)

    # ============================================================
    # 출력: GPU 비교 (모델별 × GPU별)
    # ============================================================
    print("\n" + "="*130)
    print("GPU PPA — per inference (1 sample)")
    print("="*130)
    print(f"{'Model':<16} {'GPU':<10} {'Runtime(ms)':>{W}} {'Energy(mJ)':>{W}} "
          f"{'TOPS':>{W}} {'TOPS/W':>{W}} {'TOPS/mm2':>{W}}")
    print("-"*130)
    for name in MODELS:
        for gpu in GPU_LIST:
            g = gpu_results[name][gpu]
            print(f"{name:<16} {gpu:<10} "
                  f"{g['total_runtime']*1e3:>{W}.4f} {g['total_energy']*1e3:>{W}.6f} "
                  f"{g['TOPS']:>{W}.4f} {g['TOPS_per_W']:>{W}.4f} {g['TOPS_per_mm2']:>{W}.4f}")
        print("-"*130)
    print("="*130)

    # ============================================================
    # 출력: TOPS/W 배율 비교 (ISAAC RRAM vs GPU)
    # ============================================================
    print("\n" + "="*110)
    print("ISAAC RRAM TOPS/W advantage over GPU")
    print("="*110)
    print(f"{'Model':<16}", end="")
    for gpu in GPU_LIST:
        print(f"{gpu:>18}", end="")
    print()
    print("-"*110)
    for name in MODELS:
        r = rram_results[name]
        print(f"{name:<16}", end="")
        for gpu in GPU_LIST:
            g = gpu_results[name][gpu]
            ratio = r['TOPS_per_W'] / g['TOPS_per_W'] if g['TOPS_per_W'] > 0 else float('inf')
            print(f"{ratio:>17.1f}x", end="")
        print()
    print("="*110)

    # ============================================================
    # 출력: TOPS/W 배율 비교 (Conv RRAM vs GPU)
    # ============================================================
    print("\n" + "="*110)
    print("Conventional RRAM TOPS/W advantage over GPU")
    print("="*110)
    print(f"{'Model':<16}", end="")
    for gpu in GPU_LIST:
        print(f"{gpu:>18}", end="")
    print()
    print("-"*110)
    for name in MODELS:
        c = conv_results[name]
        print(f"{name:<16}", end="")
        for gpu in GPU_LIST:
            g = gpu_results[name][gpu]
            ratio = c['TOPS_per_W'] / g['TOPS_per_W'] if g['TOPS_per_W'] > 0 else float('inf')
            print(f"{ratio:>17.1f}x", end="")
        print()
    print("="*110)

    # ============================================================
    # 출력: TOPS/W 배율 비교 (ISAAC vs Conv RRAM)
    # ============================================================
    print("\n" + "="*60)
    print("ISAAC RRAM TOPS/W advantage over Conventional RRAM")
    print("="*60)
    print(f"{'Model':<16} {'ISAAC TOPS/W':>16} {'Conv TOPS/W':>14} {'Ratio':>10}")
    print("-"*60)
    for name in MODELS:
        r = rram_results[name]
        c = conv_results[name]
        ratio = r['TOPS_per_W'] / c['TOPS_per_W'] if c['TOPS_per_W'] > 0 else float('inf')
        print(f"{name:<16} {r['TOPS_per_W']:>16.2f} {c['TOPS_per_W']:>14.2f} {ratio:>9.1f}x")
    print("="*60)


def run_scaling():
    """
    문장 수(N) 증가에 따른 energy/sample 비교
      ISAAC   : W_Q/W_K 1회 write amortized  → energy/sample 감소
      Conv    : 문장마다 K^T write            → energy/sample 일정 (높음)
      GPU     : write 없음                    → energy/sample 일정
    """
    cfg = MODELS[SCALING_MODEL]
    layers, d_model, seq_len = cfg['layers'], cfg['d_model'], cfg['seq_len']

    E_iw, T_iw = _isaac_write_once(layers, d_model)

    W = 20
    print("\n" + "="*105)
    print(f"Scaling Analysis — {SCALING_MODEL}  (seq_len={seq_len}, GPU={SCALING_GPU})")
    print(f"  ISAAC  one-time W_Q/W_K write : {E_iw*1e3:.2f} mJ  /  {T_iw*1e3:.3f} ms")
    print("="*105)
    print(f"{'N':>7}  {'ISAAC E/sent(mJ)':>{W}} {'Conv E/sent(mJ)':>{W}} {'GPU E/sent(mJ)':>{W}}  "
          f"{'Conv/ISAAC':>12} {'GPU/ISAAC':>10}")
    print("-"*105)

    for N in SCALING_N_LIST:
        r = compute_rram_ppa_for_model(layers, d_model, seq_len, N)
        c = calculate_conventional_rram_ppa_for_model(layers, d_model, seq_len, N)
        g = calculate_gpu_ppa_for_model(layers, d_model, seq_len, N)[SCALING_GPU]

        isaac_e = (E_iw + r['energy']) / N          # amortized write + inference
        conv_e  = c['energy'] / N                    # K^T write + inference (per sentence)
        gpu_e   = g['total_energy'] / N

        conv_ratio = conv_e / isaac_e if isaac_e > 0 else float('inf')
        gpu_ratio  = gpu_e  / isaac_e if isaac_e > 0 else float('inf')

        print(f"{N:>7}  {isaac_e*1e3:>{W}.4f} {conv_e*1e3:>{W}.4f} {gpu_e*1e3:>{W}.4f}  "
              f"{conv_ratio:>11.2f}x {gpu_ratio:>9.2f}x")

    print("="*105)
    print("  ISAAC : N 증가 → write cost amortized → energy/sent 감소")
    print("  Conv  : N 증가 → K^T write 반복      → energy/sent 일정 (높음)")
    print("  GPU   : N 증가 → write 없음           → energy/sent 일정")

    # latency/sample
    print()
    print(f"{'N':>7}  {'ISAAC lat/sent(ms)':>{W}} {'Conv lat/sent(ms)':>{W}} {'GPU lat/sent(ms)':>{W}}")
    print("-"*75)
    for N in SCALING_N_LIST:
        r = compute_rram_ppa_for_model(layers, d_model, seq_len, N)
        c = calculate_conventional_rram_ppa_for_model(layers, d_model, seq_len, N)
        g = calculate_gpu_ppa_for_model(layers, d_model, seq_len, N)[SCALING_GPU]

        isaac_t = (T_iw + r['runtime']) / N * 1e3
        conv_t  = c['runtime']          / N * 1e3
        gpu_t   = g['total_runtime']    / N * 1e3

        print(f"{N:>7}  {isaac_t:>{W}.4f} {conv_t:>{W}.4f} {gpu_t:>{W}.4f}")
    print("="*75)


if __name__ == "__main__":
    run()
    run_scaling()
