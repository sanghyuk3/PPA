"""
cal_latency.py
Overall latency calculation using Horowitz approximation
"""

import config
from formulation import calculate_chain_delay


def calculate_overall_latency():
    """
    Calculate total latency using RC delay chains
    
    Returns:
        dict: Latency breakdown
    """
    
    # ========================
    # Q path: Driver → RRAM → Integrator → ADC
    # ========================
    RC_list_Q = [
    (config.R_DRIVER, config.C_WORDLINE), # 기생 cap에 대한 RC
    (config.R_BITLINE_WIRE, config.C_INTEGRATOR_IN), # line 저항과 적분기 사이의 RC delay
    (config.R_INTEGRATOR, config.C_ADC_IN),
    ]
    
    delay_Q_chain, ramp_Q = calculate_chain_delay(RC_list_Q, config.INITIAL_RAMP)
    delay_Q_total = delay_Q_chain + config.T_READ + config.T_DRIVER + config.T_INTEGRATOR + config.T_ADC
    # delay_Q_total = config.T_READ + config.T_DRIVER + config.T_INTEGRATOR + config.T_ADC
    
    # ========================
    # K path: Driver → RRAM → Integrator → Comparator
    # ========================
    RC_list_K = [
        (config.R_DRIVER, config.C_WORDLINE),      # 기생 cap에 대한 RC    
        (config.R_BITLINE_WIRE, config.C_INTEGRATOR_IN),       # line 저항과 적분기 사이의 RC delay    
        (config.R_INTEGRATOR, config.C_COMPARATOR_IN),  # 적분기와 비교기 사이의 RC delay
    ]
    
    delay_K_chain, ramp_K = calculate_chain_delay(RC_list_K, config.INITIAL_RAMP)
    delay_K_total = delay_K_chain + config.T_READ + config.T_DRIVER + config.T_INTEGRATOR + config.T_COMPARATOR
    # delay_K_total = config.T_READ + config.T_DRIVER + config.T_INTEGRATOR + config.T_COMPARATOR
    
    # ========================
    # Total runtime
    # ========================
    total_q_ops = (config.RRAM_NUM_SAMPLES * config.RRAM_LAYERS * 
                   config.Q_READ_MULTIPLIER)
    total_k_ops = (config.RRAM_NUM_SAMPLES * config.RRAM_LAYERS * 
                   config.K_READ_MULTIPLIER)
    
    t_total_Q = delay_Q_total * total_q_ops
    t_total_K = delay_K_total * total_k_ops
    t_total = t_total_Q + t_total_K
    
    return {
        'delay_Q_per_op': delay_Q_total,
        'delay_K_per_op': delay_K_total,
        'delay_Q_chain': delay_Q_chain,
        'delay_K_chain': delay_K_chain,
        'ramp_Q_final': ramp_Q,
        'ramp_K_final': ramp_K,
        't_total_Q': t_total_Q,
        't_total_K': t_total_K,
        't_total': t_total,
        'total_q_ops': total_q_ops,
        'total_k_ops': total_k_ops
    }


def print_latency_breakdown(latency_result):
    """Print detailed latency breakdown"""
    
    print("\n" + "="*80)
    print("Latency Breakdown")
    print("="*80)
    
    print(f"\nQ Path (per operation):")
    print(f"  Chain delay (Driver+RRAM+Integrator): {latency_result['delay_Q_chain']*1e9:.3f} ns")
    print(f"  Total per operation: {latency_result['delay_Q_per_op']*1e9:.3f} ns")
    print(f"  Output slew rate: {latency_result['ramp_Q_final']/1e9:.3f} V/ns")

    
    print(f"\nK Path (per operation):")
    print(f"  Chain delay (Driver+RRAM+Integrator): {latency_result['delay_K_chain']*1e9:.3f} ns")
    print(f"  Total per operation: {latency_result['delay_K_per_op']*1e9:.3f} ns")
    print(f"  Output slew rate: {latency_result['ramp_K_final']/1e9:.3f} V/ns")
    
    print(f"\nTotal Runtime:")
    print(f"  Q path: {latency_result['t_total_Q']*1e3:.3f} ms ({latency_result['total_q_ops']:,} ops)")
    print(f"  K path: {latency_result['t_total_K']*1e3:.3f} ms ({latency_result['total_k_ops']:,} ops)")
    print(f"  Combined: {latency_result['t_total']*1e3:.3f} ms")
    print("="*80 + "\n")


if __name__ == "__main__":
    # Test
    print("Testing cal_latency.py")
    
    latency_result = calculate_overall_latency()
    print_latency_breakdown(latency_result)