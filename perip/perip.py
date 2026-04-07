"""
perip.py
Peripheral circuits energy calculation only
(Area is calculated separately in cal_area.py)
"""

import config
from perip.Driver import calculate_driver_energy, calculate_driver_leakage_energy
from perip.Integrator import calculate_integrator_energy, calculate_integrator_leakage_energy
from perip.ADC import calculate_adc_energy, calculate_adc_leakage_energy
from perip.Comparator import calculate_comparator_energy, calculate_comparator_leakage_energy


def calculate_energy_perip(num_input_activations, num_output_conversions, 
                          total_runtime, path_type='Q'):
    """
    Calculate peripheral circuit ENERGY for Q or K path
    (Area is calculated separately)
    """

    # ========================
    # Dynamic Energy # 한 단어당
    # ========================
    
    # Driver
    E_driver_per_op = calculate_driver_energy(
        config.V_SUPPLY, 
        config.C_DRIVER_OUTPUT
    )

    E_driver_dynamic = E_driver_per_op * num_input_activations

    # Integrator
    E_integrator_per_op = calculate_integrator_energy(
        config.C_INTEGRATOR_IN,
        config.V_SUPPLY
    )

    E_integrator_dynamic =E_integrator_per_op * num_output_conversions
    
    # ADC (Q path) or Comparator (K path)
    if path_type == 'Q':
        E_converter_per_op = calculate_adc_energy(
            config.ADC_RESOLUTION, 
            config.V_SUPPLY, 
            config.C_ADC_UNIT
        )
    else:
        E_converter_per_op = calculate_comparator_energy(
            config.C_COMPARATOR_IN, 
            config.V_SUPPLY
        )
    
    # PiM: 입력 1회 인가 시 D_OUT ADC가 병렬로 동시 동작.
    # 에너지 = E_per_ADC × D_OUT × total_reads = E_per_ADC × num_output_conversions
    E_converter_dynamic = E_converter_per_op * num_output_conversions
    # Control logic
    total_ops = num_input_activations + num_output_conversions
    E_control_dynamic = config.E_CONTROL * total_ops 
    
    # ========================
    # Leakage Energy (duty-cycle: active phase only)
    # Each component leaks only during its own conversion window.
    # Defensible: clock/power gating between conversions.
    # ========================
    if path_type == 'Q':
        D_IN  = config.Q_D_IN
        D_OUT = config.Q_D_OUT
    else:
        D_IN  = config.K_D_IN
        D_OUT = config.K_D_OUT

    n_ops = num_input_activations   # total driver activations

    # Active time per component (duty-cycle adjusted)
    t_driver_active     = n_ops / D_IN  * config.T_DRIVER
    t_integrator_active = n_ops / D_IN  * config.T_INTEGRATOR
    if path_type == 'Q':
        t_converter_active = n_ops / D_IN * config.T_ADC
    else:
        t_converter_active = n_ops / D_IN * config.T_COMPARATOR

    # Driver leakage
    E_driver_leak = calculate_driver_leakage_energy(
        config.P_LEAK_DRIVER,
        t_driver_active,
        D_IN
    )

    # Integrator leakage
    E_integrator_leak = calculate_integrator_leakage_energy(
        config.P_LEAK_INTEGRATOR,
        t_integrator_active,
        D_OUT
    )

    # ADC/Comparator leakage
    if path_type == 'Q':
        E_converter_leak = calculate_adc_leakage_energy(
            config.P_LEAK_ADC,
            t_converter_active,
            D_OUT
        )
    else:
        E_converter_leak = calculate_comparator_leakage_energy(
            config.P_LEAK_COMPARATOR,
            t_converter_active,
            D_OUT
        )

    # Control leakage (전체 runtime 기준 — 제어 로직은 항상 동작)
    E_control_leak = config.P_LEAK_CONTROL * total_runtime
    
    # ========================
    # Total Energy
    # ========================
    
    E_dynamic_total = (
        E_driver_dynamic + 
        E_integrator_dynamic + 
        E_converter_dynamic + 
        E_control_dynamic
    )
    
    E_leakage_total = (
        E_driver_leak + 
        E_integrator_leak + 
        E_converter_leak + 
        E_control_leak
    )
    
    E_total = E_dynamic_total + E_leakage_total
    #E_total = E_dynamic_total 
    
    return {
        'dynamic': {
            'driver': E_driver_dynamic,
            'integrator': E_integrator_dynamic,
            'converter': E_converter_dynamic,
            'control': E_control_dynamic,
            'total': E_dynamic_total
        },
        'leakage': {
            'driver': E_driver_leak,
            'integrator': E_integrator_leak,
            'converter': E_converter_leak,
            'control': E_control_leak,
            'total': E_leakage_total
        },
        'total': E_total
    }


if __name__ == "__main__":
    print("Testing perip.py")
    print("="*60)
    
    # Dummy PPA instance for testing
    class DummyPPA:
        def calculate_runtime(self):
            return 0.288569  # 288.569 ms
    
    ppa = DummyPPA()
    
    # Test Q path
    num_input = 243906560
    num_output = 487813120
    runtime = ppa.calculate_runtime()
    
    result_Q = calculate_energy_perip(num_input, num_output, runtime, 'Q')
    print(f"\nQ Path Energy:")
    print(f"  Dynamic: {result_Q['dynamic']['total']*1e6:.3f} μJ")
    print(f"  Leakage: {result_Q['leakage']['total']*1e6:.3f} μJ")
    print(f"  Total: {result_Q['total']*1e6:.3f} μJ")
    
    # Test K path
    num_input_k = 8003520
    num_output_k = 12635200
    
    result_K = calculate_energy_perip(num_input_k, num_output_k, runtime, 'K')
    print(f"\nK Path Energy:")
    print(f"  Dynamic: {result_K['dynamic']['total']*1e6:.3f} μJ")
    print(f"  Leakage: {result_K['leakage']['total']*1e6:.3f} μJ")
    print(f"  Total: {result_K['total']*1e6:.3f} μJ")