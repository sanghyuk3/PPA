# class ISAAC_RRAM_PPA:
#     def __init__(self):
#         # ==========================================
#         # 측정 Setup
#         # ==========================================
#         self.V_read = 0.2               # 200mV (ISAAC)
#         self.G_on = 6.5e-6              # 6.5μS (YOUR MEASUREMENT)
#         self.G_off = 1.0e-6             # 1.0μS (YOUR MEASUREMENT)
#         self.I_read = (0.05 * self.G_on + 0.95 * self.G_off) * self.V_read
#         self.ADC_resolution = 7         # ISAAC uses 7-bit
                                      
#         # ==========================================
#         # 실제 array 크기
#         # ==========================================
#         self.fab_array_rows = 48        # Physical array: 48×48
#         self.fab_array_cols = 48
#         self.RRAM_width = 10e-6
#         self.cell_pitch = 3e-6    
                        
#         # ==========================================
#         # latency
#         # ==========================================
#         self.t_read = 3e-9             # 10ns (ISAAC - ideal timing)
#         self.t_write = 100e-9           # 100ns (ISAAC)
#         self.t_driver = 0.5e-9            # 0.5ns (DAC/Driver)
#         self.t_SH = 0.3e-9                # 0.3ns (Sample & Hold)
#         self.t_ADC = 10e-9                # 10ns (7-bit SAR ADC)
#         self.t_integrator = 3e-9          # 3ns (Charge integrator)
#         self.t_shifter = 0.2e-9           # 0.2ns (Shift-and-Add)
#         self.t_control = 0.5e-9           # 0.5ns (Control logic)
#         self.t_pulse_g = 0                # 0 (t_read에 포함)
#         self.t_comparator = 1.5e-9        # 1.5 ns

#         # ==========================================
#         # Energy
#         # ==========================================
#         self.E_read = self.V_read * self.I_read * self.t_read
#         self.E_write = 2e-12            # 2pJ (ISAAC)
#         self.E_ADC = 50e-15             # 50fJ per conversion
#         self.E_SH = 10e-15              # 10fJ
#         self.E_driver = 50e-15          # 50fJ (ISAAC: DAC + driver)
#         self.E_shifter = 5e-15          # 5fJ per operation
#         self.E_TDC = 100e-15            # 100fJ (ISSCC 2020 SAR-TDC)
#         self.E_integrator = 80e-15      # 80fJ
#         self.E_control = 20e-15         # 20fJ
#         self.E_pulse_g = 5e-15          # 5fJ
#         self.E_comparator = 5e-15       # 5fJ

#         # ==========================================
#         # Area
#         # ==========================================
#         self.A_ADC = 1000e-12               # 1000 μm²
#         self.A_SH = 50e-12                  # 50 μm²
#         self.A_driver = 500e-12             # 500 μm²
#         self.A_shifter = 100e-12            # 100 μm²
#         self.A_TDC = 2000e-12               # 2000 μm²
#         self.A_integrator = 1500e-12        # 1500 μm²
#         self.A_control = 1000e-12           # 1000 μm²
#         self.A_pulse_g = 20e-12             # 20 μm²
#         self.A_comparator = 10e-12          # 10 μm²
#         self.A_cell = self.cell_pitch ** 2  # 900um²

#         # ==========================================
#         # RC 파라미터 추가 (calculate_runtime용)
#         # ==========================================
#         self.R_driver = 100          # Driver output resistance (Ω)
#         self.C_integrator_in = 10e-15   # Integrator input cap (10fF)
#         self.R_integrator = 1000     # Integrator output resistance (Ω)
#         self.C_ADC_in = 5e-15           # ADC input cap (5fF)
#         self.R_comparator = 500      # Comparator output resistance (Ω)
#         self.C_comparator_in = 5e-15 # Comparator input cap (5fF)
        
#         # RRAM 파라미터
#         self.R_rram = 1.0 / self.G_on if self.G_on > 0 else 154e3  # ~154kΩ
#         self.C_bitline = 100e-15      # 100fF (bitline capacitance)
        
#         # 초기 슬루율 (외부 입력)
#         self.initial_ramp = 1e9      # 1 V/ns

"""
param.py
ISAAC_RRAM_PPA class - main PPA calculator
"""

import config
from PPA.cal_energy import calculate_overall_energy, print_energy_breakdown
from PPA.cal_latency import calculate_overall_latency, print_latency_breakdown
from PPA.cal_area import calculate_overall_area


class ISAAC_RRAM_PPA:
    def __init__(self):
        """Initialize PPA calculator"""
        # Import parameters from config
        self.V_read = config.V_READ
        self.G_on = config.G_ON
        self.G_off = config.G_OFF
        self.t_read = config.T_READ
        self.V_supply = config.V_SUPPLY
        self.ADC_resolution = config.ADC_RESOLUTION
        
        # Cache for calculated values
        self._runtime_cache = None
        self._energy_cache = None

    def get_full_results(self, verbose=True):
        # Calculate all metrics
        latency_result = calculate_overall_latency()
        runtime = latency_result['t_total']
        energy_result = calculate_overall_energy(runtime)
        area_result = calculate_overall_area()

        energy = energy_result['E_total']
        mac_ops = energy_result['total_mac_ops']

        # Derived metrics
        power = energy / runtime if runtime > 0 else 0
        tops = (mac_ops * 2) / runtime / 1e12 if runtime > 0 else 0  # MAC = 2 OPS
        area_mm2 = area_result['A_total'] * 1e6
        tops_per_w = tops / power if power > 0 else 0
        tops_per_mm2 = tops / area_mm2 if area_mm2 > 0 else 0
        energy_per_sample = energy / config.RRAM_NUM_SAMPLES

        results = {
            'runtime': runtime,
            'energy': energy,
            'power': power,
            'area': area_mm2,
            'mac_ops': mac_ops,
            'TOPS': tops,
            'TOPS_per_W': tops_per_w,
            'TOPS_per_mm2': tops_per_mm2,
            'energy_per_sample': energy_per_sample,
            'latency_breakdown': latency_result,
            'energy_breakdown': energy_result
        }

        if verbose:
            print_latency_breakdown(latency_result)
            print_energy_breakdown(energy_result)
            self._print_summary(results)

        return results
    
    def _print_summary(self, results):
        """Print summary metrics"""
        print("="*80)
        print("PPA Summary")
        print("="*80)
        print(f"Runtime:          {results['runtime']*1e3:.3f} ms")
        print(f"Energy:           {results['energy']*1e3:.6f} mJ")
        print(f"Power:            {results['power']*1e3:.4f} mW")
        print(f"Area:             {results['area']:.6f} mm²")
        print(f"MAC Operations:   {results['mac_ops']/1e9:.2f} GMAC")
        print(f"TOPS:             {results['TOPS']:.6f}")
        print(f"TOPS/W:           {results['TOPS_per_W']:.4f}")
        print(f"TOPS/mm²:         {results['TOPS_per_mm2']:.6f}")
        print(f"Energy/sample:    {results['energy_per_sample']*1e6:.3f} μJ")
        print("="*80 + "\n")


def compute_rram_ppa_for_model(layers, d_model, sent_len, num_samples):
    """
    임의의 Transformer 모델에 대해 RRAM PPA 계산.
    config 전역변수를 임시로 override 후 복구 (try/finally 보장).

    Args:
        layers     (int): Transformer layer 수
        d_model    (int): Hidden dimension (Q_D_IN = K_D_IN = Q_D_OUT = K_D_OUT)
        sent_len   (int): 시퀀스 길이 (RRAM_SENT_LEN)
        num_samples(int): 평가 샘플 수 (RRAM_NUM_SAMPLES)

    Returns:
        dict: get_full_results() 반환값과 동일
    """
    KEYS = ['RRAM_LAYERS', 'Q_D_IN', 'Q_D_OUT', 'K_D_IN', 'K_D_OUT',
            'RRAM_SENT_LEN', 'RRAM_NUM_SAMPLES',
            'Q_READ_MULTIPLIER', 'K_READ_MULTIPLIER']
    saved = {k: getattr(config, k) for k in KEYS}
    try:
        config.RRAM_LAYERS        = layers
        config.Q_D_IN             = d_model
        config.Q_D_OUT            = d_model
        config.K_D_IN             = d_model
        config.K_D_OUT            = d_model
        config.RRAM_SENT_LEN      = sent_len
        config.RRAM_NUM_SAMPLES   = num_samples
        config.Q_READ_MULTIPLIER  = sent_len * 72
        config.K_READ_MULTIPLIER  = sent_len
        return ISAAC_RRAM_PPA().get_full_results(verbose=False)
    finally:
        for k, v in saved.items():
            setattr(config, k, v)


if __name__ == "__main__":
    print("Testing ISAAC_RRAM_PPA")
    config.print_config()

    ppa = ISAAC_RRAM_PPA()
    results = ppa.get_full_results(verbose=True)