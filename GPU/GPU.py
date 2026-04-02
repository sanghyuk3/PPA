import config

def calculate_gpu_qkt_operation():
    """Calculate PPA for all GPUs (uses current config globals)."""
    gpu_list = ['V100', 'A100', 'H100', 'RTX_4090', 'RTX_3090']
    return {g: _calculate_single_gpu(g) for g in gpu_list}


def calculate_gpu_ppa_for_model(layers, d_model, sent_len, num_samples):
    """
    임의의 Transformer 모델에 대해 모든 GPU의 PPA 계산.
    config 전역변수를 임시 override 후 복구.

    Returns:
        dict[gpu_name] = _calculate_single_gpu() 결과
    """
    KEYS = ['GPU_NUM_LAYERS', 'GPU_D_MODEL', 'GPU_MAX_SENT_LEN', 'GPU_NUM_SAMPLES']
    saved = {k: getattr(config, k) for k in KEYS}
    try:
        config.GPU_NUM_LAYERS   = layers
        config.GPU_D_MODEL      = d_model
        config.GPU_MAX_SENT_LEN = sent_len
        config.GPU_NUM_SAMPLES  = num_samples
        return calculate_gpu_qkt_operation()
    finally:
        for k, v in saved.items():
            setattr(config, k, v)

def _calculate_single_gpu(gpu_name):

    # gpu_specs = {
    #     'V100': {
    #         'int8_tops': 125,
    #         'energy_per_mac': 0.5e-12,      # 0.5 pJ/MAC (INT8)
    #         'die_size': 815,
    #         'memory_bandwidth': 900e9,
    #         'dram_energy_per_byte': 20e-12,
    #     },
    #     'A100': {
    #         'int8_tops': 624,
    #         'energy_per_mac': 0.3e-12,      # 0.3 pJ/MAC (더 효율적)
    #         'die_size': 826,
    #         'memory_bandwidth': 2039e9,
    #         'dram_energy_per_byte': 15e-12,
    #     },
    #     'H100': {
    #         'int8_tops': 3958,
    #         'energy_per_mac': 0.2e-12,      # 0.2 pJ/MAC (최신 세대)
    #         'die_size': 814,
    #         'memory_bandwidth': 3350e9,
    #         'dram_energy_per_byte': 12e-12,
    #     },
    #     'RTX_4090': {
    #         'int8_tops': 660.6,                  # ✅ 660.6 TOPS INT8 dense (공식 whitepaper)
    #         'energy_per_mac': 0.4e-12,           # 0.4 pJ/MAC (4nm, gaming GPU)
    #         'die_size': 609,                     # ✅ 608.4 mm² (공식)
    #         'memory_bandwidth': 1008e9,          # ✅ 1008 GB/s GDDR6X (공식)
    #         'dram_energy_per_byte': 30e-12,      # 25-30 pJ/byte (GDDR6X, HBM보다 비효율)
    #     },
    #     'RTX_3090': {
    #         'int8_tops': 284,                    # ✅ 284 TOPS INT8 dense (공식)
    #         'energy_per_mac': 0.6e-12,           # 0.6 pJ/MAC (8nm 구세대)
    #         'die_size': 628,                     # ✅ 628 mm² (GA102)
    #         'memory_bandwidth': 936e9,           # ✅ 936 GB/s GDDR6X (공식)
    #         'dram_energy_per_byte': 35e-12,      # 30-35 pJ/byte (구세대 GDDR6X)
    #     },
    # }

    gpu_specs = { # DRAM은 HBM3 기준
        'V100': { # V100 NVLink 기준
            'int8_tops': 125,
            'tdp': 300,        # 0.5 pJ/MAC
            'die_size': 815,
            'memory_bandwidth': 900e9,
            'dram_energy_per_byte': 10e-12,      # HBM2
            'dram_latency': 100e-9,  
            'GPU_PIPELINE_LATENCY' : 50e-9
        },
        'A100': { # A100 80GB PCle 기준
            'int8_tops': 624,    
            'tdp': 300,         
            'die_size': 826,
            'memory_bandwidth': 1935e9,          # 1935 GB/s (80GB 모델)
            'dram_energy_per_byte': 10e-12,      # HBM2e
            'dram_latency': 100e-9,  
            'GPU_PIPELINE_LATENCY' : 30e-9
        },
        'H100': { # H100 SXM 기준
            'int8_tops': 3958,
            'tdp': 700,            
            'die_size': 814,
            'memory_bandwidth': 3350e9,          # 3.35 TB/s
            'dram_energy_per_byte': 9e-12,      # HBM3
            'dram_latency': 80e-9,  
            'GPU_PIPELINE_LATENCY' : 20e-9
        },
        'RTX_4090': { # 4090 FP16 기준
            'int8_tops': 1321,
            'tdp': 450,           # 0.4 pJ/MAC
            'die_size': 609,
            'memory_bandwidth': 1008e9,
            'dram_energy_per_byte': 15e-12,      # GDDR6X
            'dram_latency': 120e-9,  
            'GPU_PIPELINE_LATENCY' : 40e-9
        },
        'RTX_3090': {
            'int8_tops': 284,
            'tdp' : 350,           # 0.6 pJ/MAC
            'die_size': 628,
            'memory_bandwidth': 936e9,
            'dram_energy_per_byte': 18e-12,      # GDDR6X
            'dram_latency': 120e-9,  
            'GPU_PIPELINE_LATENCY' : 50e-9 
        },
    }
    spec = gpu_specs[gpu_name]

    # ====================================
    # Size (문장 하나)
    # ====================================
    # W_K_size = config.GPU_D_MODEL * config.GPU_D_MODEL 
    # W_Q_size = config.GPU_D_MODEL * config.GPU_D_MODEL 

    X_size = config.GPU_MAX_SENT_LEN * config.GPU_D_MODEL  
    Q_size = config.GPU_MAX_SENT_LEN * config.GPU_D_MODEL 
    K_size = config.GPU_MAX_SENT_LEN * config.GPU_D_MODEL 
    QKT_size = config.GPU_MAX_SENT_LEN * config.GPU_MAX_SENT_LEN

    # ===================================
    # MAC
    # ===================================
    Q_mac_ops = config.GPU_NUM_SAMPLES * config.GPU_MAX_SENT_LEN * config.GPU_D_MODEL * config.GPU_D_MODEL * config.GPU_NUM_LAYERS
    K_mac_ops = config.GPU_NUM_SAMPLES * config.GPU_MAX_SENT_LEN * config.GPU_D_MODEL * config.GPU_D_MODEL * config.GPU_NUM_LAYERS
    QKT_mac_ops = config.GPU_NUM_SAMPLES * config.GPU_MAX_SENT_LEN * config.GPU_MAX_SENT_LEN * config.GPU_D_MODEL * config.GPU_NUM_LAYERS
    total_mac_ops = Q_mac_ops + K_mac_ops + QKT_mac_ops
    
    # ===================================
    # runtime (Array에서도 W write시간은 안보기 떄문에, W load time 제거)
    # ===================================
    # W_Q_load_time = W_Q_size / spec['memory_bandwidth']
    # W_K_load_time = W_K_size / spec['memory_bandwidth']

    X_load_time = X_size * config.GPU_NUM_SAMPLES * config.GPU_NUM_LAYERS / spec['memory_bandwidth'] + \
                  spec['dram_latency'] * config.GPU_NUM_LAYERS  
    Q_load_time = Q_size * config.GPU_NUM_SAMPLES * config.GPU_NUM_LAYERS / spec['memory_bandwidth']+ \
                  spec['dram_latency'] * config.GPU_NUM_LAYERS  
    K_load_time = K_size * config.GPU_NUM_SAMPLES * config.GPU_NUM_LAYERS / spec['memory_bandwidth']+ \
                  spec['dram_latency'] * config.GPU_NUM_LAYERS  

    Q_write_time = Q_size * config.GPU_NUM_SAMPLES * config.GPU_NUM_LAYERS / spec['memory_bandwidth']+ \
                  spec['dram_latency'] * config.GPU_NUM_LAYERS  
    K_write_time = K_size * config.GPU_NUM_SAMPLES * config.GPU_NUM_LAYERS / spec['memory_bandwidth']+ \
                  spec['dram_latency'] * config.GPU_NUM_LAYERS  
    # QKT_write_time = QKT_size * config.GPU_NUM_SAMPLES * config.GPU_NUM_LAYERS / spec['memory_bandwidth']

    Q_compute_time = (Q_mac_ops) / (spec['int8_tops'] * 1e12)+ \
                     spec['GPU_PIPELINE_LATENCY']* config.GPU_NUM_LAYERS
    K_compute_time = (K_mac_ops) / (spec['int8_tops'] * 1e12)+ \
                     spec['GPU_PIPELINE_LATENCY']* config.GPU_NUM_LAYERS
    QKT_compute_time = (QKT_mac_ops) / (spec['int8_tops'] * 1e12)+ \
                     spec['GPU_PIPELINE_LATENCY']* config.GPU_NUM_LAYERS

    # ===================================
    # Total runtime
    # ===================================
    # weight_load_time = W_K_load_time + W_Q_load_time
    activation_time = 2 * X_load_time + Q_load_time + K_load_time + Q_write_time + K_write_time 
    compute_time = Q_compute_time + K_compute_time + QKT_compute_time
    total_runtime = activation_time + compute_time

    # ===================================
    # Energy
    # ===================================
    Activation_bytes_per_sentence = X_size * 2 +  Q_size * 2 +  K_size * 2
    total_memory_bytes = Activation_bytes_per_sentence * config.GPU_NUM_SAMPLES * config.GPU_NUM_LAYERS
    memory_energy = total_memory_bytes * spec['dram_energy_per_byte']
    
    # compute energy
    compute_energy = spec['tdp'] * compute_time

    total_dram_latency_time = (72) * spec['dram_latency']  # x,Q,K 총 6을 12layer
    dram_idle_energy = spec['tdp'] * 0.15 * total_dram_latency_time  # 15% TDP

    total_pipeline_latency_time = spec['GPU_PIPELINE_LATENCY'] * 3 * config.GPU_NUM_LAYERS  # Q, K, QKT
    pipeline_idle_energy = spec['tdp'] * 0.3 * total_pipeline_latency_time  # 30% TDP

    total_energy = memory_energy + compute_energy + dram_idle_energy + pipeline_idle_energy
    
    # ===================================
    # 지표
    # ===================================
    power = total_energy / total_runtime 
    tops = total_mac_ops / total_runtime / 1e12 
    tops_per_w = tops / power if power > 0 else 0
    area_mm2 = spec['die_size']
    tops_per_mm2 = tops / area_mm2 if area_mm2 > 0 else 0
    energy_per_sentence = total_energy / config.GPU_NUM_SAMPLES
    area_per_sentence = area_mm2 / config.GPU_NUM_SAMPLES

    return {
        'platform': gpu_name,
        'total_runtime': total_runtime,
        'power' : power,
        'compute_time': compute_time,
        'total_energy': total_energy,
        'compute_energy': compute_energy,
        'memory_energy': memory_energy,
        'mac_ops': total_mac_ops,  
        'area': area_mm2,
        'TOPS': tops,
        'TOPS_per_W': tops_per_w,
        'TOPS_per_mm2': tops_per_mm2,
        'energy_per_sentence' : energy_per_sentence,
        'area_per_sentence' : area_per_sentence
    }