"""
config.py
Global configuration for RRAM simulation
"""

# ============================================================
# RRAM Architecture
# ============================================================
RRAM_LAYERS = 12
RRAM_NUM_SAMPLES = 872
RRAM_SENT_LEN = 36

# Q matrix dimensions
Q_D_IN = 15
Q_D_OUT = 30

# K matrix dimensions
K_D_IN = 19
K_D_OUT = 30

a = 256

Q_D_IN = a
Q_D_OUT = a

# K matrix dimensions
K_D_IN = a
K_D_OUT = a

# Derived parameters
Q_READ_MULTIPLIER = RRAM_SENT_LEN * RRAM_SENT_LEN  # 1296
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
G_ON = 6.5e-6            # 6.5 μS
G_OFF = 1.0e-6           # 1.0 μS
ADC_RESOLUTION = 7        # 7-bit

# Technology
TECH_NODE = 65e-9        # 65nm
#CELL_PITCH = 3e-6        # 3μm
CELL_PITCH = 30e-6        # 3μm

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
T_INTEGRATOR = 5e-9 

# ADC
C_ADC_UNIT = 1e-15            # 1fF (SAR ADC unit cap)
C_ADC_IN = 5e-15              # 5fF (ADC input cap)
T_ADC = 90e-9                 # 10ns
P_LEAK_ADC = 5e-6             # 5μW

# Comparator
C_COMPARATOR_IN = 5e-15       # 5fF
T_COMPARATOR = 1.5e-9         # 1.5ns
P_LEAK_COMPARATOR = 1e-6      # 1μW

# Control Logic
E_CONTROL = 20e-15            # 20fJ per operation
P_LEAK_CONTROL = 0.5e-6       # 0.5μW

# ============================================================
# RRAM Cell Timing
# ============================================================
# T_READ = 5e-9  
T_READ = 120e-6   
T_WRITE = 100e-9              # 100ns
T_PULSE_G = 0                 # Included in T_READ
T_DRIVER = 0.5e-9             # 0.5ns

# ============================================================
# RC Parameters (for Horowitz)
# ============================================================
R_RRAM = 1.0 / G_ON if G_ON > 0 else 154e3  # ~154kΩ  
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