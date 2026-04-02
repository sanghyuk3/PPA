"""
cal_energy.py
Overall energy calculation integrating RRAM cells and peripherals
"""

import config
from Cell import calculate_rram_read_energy_gavg
from perip.perip import calculate_energy_perip


def calculate_overall_energy(runtime):
    """
    Calculate total energy consumption
    
    Args:
        ppa_instance: ISAAC_RRAM_PPA instance (needs calculate_runtime method)
    
    Returns:
        dict: Energy breakdown
    """
    
    # ========================
    # Total MAC operations
    # ========================
    total_q_macs = (config.RRAM_NUM_SAMPLES * config.RRAM_LAYERS * 
                    config.Q_READ_MULTIPLIER * config.Q_D_IN * config.Q_D_OUT)
    total_k_macs = (config.RRAM_NUM_SAMPLES * config.RRAM_LAYERS * 
                    config.K_READ_MULTIPLIER * config.K_D_IN * config.K_D_OUT)
    total_mac_ops = total_q_macs + total_k_macs
    
    # ========================
    # 1. RRAM Cell Read Energy (측정된 G_AVG 사용, Q/K 별도 적용)
    # ========================
    E_rram_cell_Q = calculate_rram_read_energy_gavg(config.V_READ, config.G_AVG_Q, config.T_READ)
    E_rram_cell_K = calculate_rram_read_energy_gavg(config.V_READ, config.G_AVG_K, config.T_READ)

    E_RRAM_total = total_q_macs * E_rram_cell_Q + total_k_macs * E_rram_cell_K

    # ========================
    # 2. Peripheral Circuit Energy
    # ========================
    total_runtime = runtime # inference에 걸리는 시간 // leakage 계산에 필수!
    # total_runtime = 0.00001
    
    # Q path
    num_Q_input = total_q_macs // config.Q_D_OUT
    num_Q_output = total_q_macs // config.Q_D_IN
    
    result_Q = calculate_energy_perip(
        num_Q_input, 
        num_Q_output,
        total_runtime, 
        'Q'
    )
    
    # K path
    num_K_input = total_k_macs // config.K_D_OUT
    num_K_output = total_k_macs // config.K_D_IN
    
    result_K = calculate_energy_perip(
        num_K_input, 
        num_K_output, 
        total_runtime, 
        'K'
    )
    
    E_perip_total = result_Q['total'] + result_K['total']
    
    # ========================
    # Total Energy
    # ========================
    E_total = E_RRAM_total + E_perip_total
    
    # ========================
    # Return breakdown
    # ========================
    return {
        'E_RRAM': E_RRAM_total,
        'E_perip_Q': result_Q,
        'E_perip_K': result_K,
        'E_perip_total': E_perip_total,
        'E_total': E_total,
        'total_mac_ops': total_mac_ops,
        'total_runtime': total_runtime
    }


def print_energy_breakdown(energy_result):
    """Print detailed energy breakdown"""
    
    E_rram = energy_result['E_RRAM']
    result_Q = energy_result['E_perip_Q']
    result_K = energy_result['E_perip_K']
    E_total = energy_result['E_total']
    
    print("\n" + "="*80)
    print("Energy Breakdown")
    print("="*80)
    print(f"{'Component':<30} {'Dynamic (μJ)':<15} {'Leakage (μJ)':<15} {'Total (μJ)':<15}")
    print("-"*80)
    
    # RRAM
    print(f"{'RRAM Cells':<30} {E_rram*1e3:<15.6f} {0:<15.6f} {E_rram*1e3:<15.6f}")
    
    # Q path
    print(f"{'Q Path:':<30}")
    print(f"{'  Driver':<30} {result_Q['dynamic']['driver']*1e6:<15.6f} "
          f"{result_Q['leakage']['driver']*1e6:<15.6f} "
          f"{(result_Q['dynamic']['driver']+result_Q['leakage']['driver'])*1e6:<15.6f}")
    print(f"{'  Integrator':<30} {result_Q['dynamic']['integrator']*1e6:<15.6f} "
          f"{result_Q['leakage']['integrator']*1e6:<15.6f} "
          f"{(result_Q['dynamic']['integrator']+result_Q['leakage']['integrator'])*1e6:<15.6f}")
    print(f"{'  ADC':<30} {result_Q['dynamic']['converter']*1e6:<15.6f} "
          f"{result_Q['leakage']['converter']*1e6:<15.6f} "
          f"{(result_Q['dynamic']['converter']+result_Q['leakage']['converter'])*1e6:<15.6f}")
    
    # K path
    print(f"{'K Path:':<30}")
    print(f"{'  Driver':<30} {result_K['dynamic']['driver']*1e6:<15.6f} "
          f"{result_K['leakage']['driver']*1e6:<15.6f} "
          f"{(result_K['dynamic']['driver']+result_K['leakage']['driver'])*1e6:<15.6f}")
    print(f"{'  Integrator':<30} {result_K['dynamic']['integrator']*1e6:<15.6f} "
          f"{result_K['leakage']['integrator']*1e6:<15.6f} "
          f"{(result_K['dynamic']['integrator']+result_K['leakage']['integrator'])*1e6:<15.6f}")
    print(f"{'  Comparator':<30} {result_K['dynamic']['converter']*1e6:<15.6f} "
          f"{result_K['leakage']['converter']*1e6:<15.6f} "
          f"{(result_K['dynamic']['converter']+result_K['leakage']['converter'])*1e6:<15.6f}")
    
    print("-"*80)
    
    E_dynamic_total = (result_Q['dynamic']['total'] + result_K['dynamic']['total'] + E_rram)
    E_leakage_total = (result_Q['leakage']['total'] + result_K['leakage']['total'])
    
    print(f"{'Dynamic Total':<30} {E_dynamic_total*1e6:<15.6f}")
    print(f"{'Leakage Total':<30} {'':<15} {E_leakage_total*1e6:<15.6f} {E_leakage_total*1e6:<15.6f}")
    print(f"{'TOTAL':<30} {'':<15} {'':<15} {E_total*1e6:<15.6f}")
    print("="*80 + "\n")


if __name__ == "__main__":
    # Test
    from param import ISAAC_RRAM_PPA
    
    print("Testing cal_energy.py")
    ppa = ISAAC_RRAM_PPA()
    
    energy_result = calculate_overall_energy(ppa)
    print_energy_breakdown(energy_result)
    
    print(f"Total MAC operations: {energy_result['total_mac_ops']/1e9:.2f} GMAC")
    print(f"Total runtime: {energy_result['total_runtime']*1e3:.3f} ms")
    print(f"Total energy: {energy_result['E_total']*1e3:.6f} mJ")