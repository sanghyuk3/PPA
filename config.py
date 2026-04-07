"""
config.py
Global configuration for RRAM simulation
"""

# ============================================================
# RRAM Architecture
# ============================================================
RRAM_LAYERS = 12
RRAM_NUM_SAMPLES = 872
RRAM_SENT_LEN = 36*14

# Q matrix dimensions (실제 W_Q: 768×768, 24×24 측정 어레이에서 확장)
Q_D_IN = 768
Q_D_OUT = 768

# K matrix dimensions (실제 W_K: 768×768)
K_D_IN = 768
K_D_OUT = 768

# Physical array size (measured device: 24×48 physical = 24×24 logical, dual-rail)
PHYS_ROWS = 24
PHYS_COLS = 24   # logical cols (physical 48 = pos + neg dual-rail)

# Derived parameters
_K_LENGTH = RRAM_SENT_LEN // 14          # 실제 seq_len (K 벡터 개수)
Q_READ_MULTIPLIER = RRAM_SENT_LEN * _K_LENGTH  # = seq_len² × 14
K_READ_MULTIPLIER = RRAM_SENT_LEN

# ============================================================
# GPU Architecture
# ============================================================
GPU_NUM_LAYERS = 12
GPU_D_MODEL = 768
GPU_MAX_SENT_LEN = 36
GPU_NUM_SAMPLES = 872

# ============================================================
# Hardware Parameters
# ============================================================
# ISAAC-based RRAM cell parameters
V_READ = 0.2              # 200mV
G_ON = 6.5e-6            # 6.5 μS = 1.3e-6 A
G_OFF = 1.0e-7           # 0.1 μS (HRS, G_OFF < G_AVG_Q=0.564μS)
ADC_RESOLUTION = 7        # 7-bit

# Technology
TECH_NODE = 65e-9        # 65nm
CELL_PITCH = 3e-6        # 3μm
#CELL_PITCH = 30e-6        # 30μm

V_SUPPLY = 1.0           # 1V, ADC나 이런 곳에서는 1.0V로 함.

# ============================================================
# Component Parameters
# ============================================================
# Driver
C_DRIVER_OUTPUT = 20e-15      # 20fF
R_DRIVER = 100                # 100Ω
P_LEAK_DRIVER = 1e-6          # 1μW

# Integrator
C_INTEGRATOR_IN = 10e-15      # 10fF
R_INTEGRATOR = 1000           # 1kΩ
P_LEAK_INTEGRATOR = 2e-6      # 2μW
T_INTEGRATOR = 1e-9

# ADC
C_ADC_UNIT = 1e-15            # 1fF (SAR ADC unit cap)
C_ADC_IN = 5e-15              # 5fF (ADC input cap)
T_ADC = 6e-9                  # 6ns (ISAAC 65nm SAR ADC 기준)
P_LEAK_ADC = 5e-6             # 5μW

# Comparator
C_COMPARATOR_IN = 5e-15       # 5fF
T_COMPARATOR = 0.5e-9         # 0.5ns (ideal regenerative comparator)
P_LEAK_COMPARATOR = 1e-6      # 1μW

# Control Logic
E_CONTROL = 20e-15            # 20fJ per operation
P_LEAK_CONTROL = 0.5e-6       # 0.5μW

# ============================================================
# RRAM Cell Timing
# ============================================================
T_READ = 10e-9                 # Ideal read time
# T_READ = 120e-6             # Real read time
T_WRITE = 100e-9              # 100ns
T_PULSE_G = 0                 # Included in T_READ
T_DRIVER = 0.5e-9             # 0.5ns

# ============================================================
# RC Parameters (for Horowitz)
# ============================================================
# Measured average conductance (24×24 array → 768×768 full matrix, same per-cell stats)
G_AVG_Q = 5.64245e-7     # W_Q 평균 conductance (S)
G_AVG_K = 5.644299e-7    # W_K 평균 conductance (S)
R_RRAM_Q = 1.0 / G_AVG_Q
R_RRAM_K = 1.0 / G_AVG_K

# Programming variation / tuning error (SW-matched, 24×24 측정, per-cell i.i.d.)
VARIATION_MEAN_Q =  4.60e-4   # W_Q 평균 오차
VARIATION_STD_Q  =  5.66e-3   # W_Q 표준편차
VARIATION_MEAN_K = -5.27e-4   # W_K 평균 오차
VARIATION_STD_K  =  4.39e-2   # W_K 표준편차

# Legacy average (기존 수식 호환용)
G_AVG = G_ON * 0.1 + G_OFF * 0.9
R_RRAM = 1.0 / G_AVG if G_AVG > 0 else 154e3
C_WORDLINE = 5e-12
C_RRAM_IN = 1e-12
C_BITLINE = 12.8e-12 
R_BITLINE_WIRE = 50                    
INITIAL_RAMP = 1e9                           # 1V/ns

# ============================================================
# Helper Functions
# ============================================================
def print_config():
    """Print current configuration"""
    print("\n" + "="*70)
    print("RRAM Simulation Configuration")
    print("="*70)
    print(f"{'Parameter':<30} {'Value':<40}")
    print("-"*70)
    print(f"{'Layers':<30} {RRAM_LAYERS}")
    print(f"{'Num samples':<30} {RRAM_NUM_SAMPLES}")
    print(f"{'Sentence length':<30} {RRAM_SENT_LEN}")
    print(f"{'Q matrix':<30} {Q_D_IN} × {Q_D_OUT}")
    print(f"{'K matrix':<30} {K_D_IN} × {K_D_OUT}")
    print(f"{'V_supply':<30} {V_SUPPLY} V")
    print("="*70 + "\n")

if __name__ == "__main__":
    print_config()