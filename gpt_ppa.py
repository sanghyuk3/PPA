"""
gpt_ppa.py
GPT-2 variant RRAM PPA vs GPU comparison.
GPT-2와 BERT는 동일한 Transformer QKV 구조이므로 같은 RRAM 하드웨어 적용 가능.
"""

# ============================================================
# GPT-2 Model Configurations
# ============================================================
GPT_MODELS = {
    'GPT-2 Small':  {'layers': 12, 'd_model': 768,  'num_heads': 12},
    'GPT-2 Medium': {'layers': 24, 'd_model': 1024, 'num_heads': 16},
    'GPT-2 Large':  {'layers': 36, 'd_model': 1280, 'num_heads': 20},
    'GPT-2 XL':     {'layers': 48, 'd_model': 1600, 'num_heads': 25},
}

# 평가 조건 (BERT와 동일 workload 기준)
SEQ_LEN     = 128   # 추론 시 typical 시퀀스 길이
NUM_SAMPLES = 872   # BERT GLUE 평가와 동일


def run_gpt_ppa_comparison():
    """
    각 GPT-2 모델 크기별 RRAM PPA와 GPU PPA를 계산한다.
    Returns: dict[model_name] = {'rram': ..., 'gpu': ..., 'cfg': ...}
    """
    from param import compute_rram_ppa_for_model
    from GPU.GPU import calculate_gpu_ppa_for_model

    results = {}
    for model_name, mcfg in GPT_MODELS.items():
        print(f"  Computing {model_name} "
              f"(L={mcfg['layers']}, d={mcfg['d_model']})...", flush=True)

        rram = compute_rram_ppa_for_model(
            layers=mcfg['layers'],
            d_model=mcfg['d_model'],
            sent_len=SEQ_LEN,
            num_samples=NUM_SAMPLES,
        )
        gpu = calculate_gpu_ppa_for_model(
            layers=mcfg['layers'],
            d_model=mcfg['d_model'],
            sent_len=SEQ_LEN,
            num_samples=NUM_SAMPLES,
        )
        results[model_name] = {'rram': rram, 'gpu': gpu, 'cfg': mcfg}

    return results


def print_gpt_comparison(results):
    GPU_ORDER = ['V100', 'A100', 'H100', 'RTX_3090', 'RTX_4090']
    W = 14

    print("\n" + "=" * 130)
    print(f"GPT-2 RRAM PPA vs GPU  (seq_len={SEQ_LEN}, n_samples={NUM_SAMPLES})")
    print("=" * 130)

    for model_name, r in results.items():
        rram = r['rram']
        gpu  = r['gpu']
        cfg  = r['cfg']

        header = (f"  {model_name}  "
                  f"(layers={cfg['layers']}, d_model={cfg['d_model']}, "
                  f"heads={cfg['num_heads']})")
        print(f"\n{header}")
        print(f"  {'Metric':<22} {'RRAM':<{W}}", end='')
        for g in GPU_ORDER:
            print(f" {g:<{W}}", end='')
        print()
        print("  " + "-" * (22 + W + W * len(GPU_ORDER) + len(GPU_ORDER)))

        def row(label, rram_val, gpu_vals, fmt='.3f'):
            print(f"  {label:<22} {format(rram_val, fmt):<{W}}", end='')
            for g in GPU_ORDER:
                print(f" {format(gpu_vals[g], fmt):<{W}}", end='')
            print()

        # Runtime
        row('Runtime (ms)',
            rram['runtime'] * 1e3,
            {g: gpu[g]['total_runtime'] * 1e3 for g in GPU_ORDER})

        # Speedup vs V100
        v100_rt = gpu['V100']['total_runtime']
        row('Speedup vs V100 (×)',
            v100_rt / rram['runtime'],
            {g: v100_rt / gpu[g]['total_runtime'] for g in GPU_ORDER}, fmt='.2f')

        # Energy
        row('Energy (mJ)',
            rram['energy'] * 1e3,
            {g: gpu[g]['total_energy'] * 1e3 for g in GPU_ORDER}, fmt='.4f')

        # Energy reduction vs V100
        v100_e = gpu['V100']['total_energy']
        row('Reduction vs V100 (×)',
            v100_e / rram['energy'],
            {g: v100_e / gpu[g]['total_energy'] for g in GPU_ORDER}, fmt='.2f')

        # TOPS/W
        row('TOPS/W',
            rram['TOPS_per_W'],
            {g: gpu[g]['TOPS_per_W'] for g in GPU_ORDER}, fmt='.4f')

        # TOPS/mm²
        row('TOPS/mm²',
            rram['TOPS_per_mm2'],
            {g: gpu[g]['TOPS_per_mm2'] for g in GPU_ORDER}, fmt='.4f')

    print("\n" + "=" * 130)
