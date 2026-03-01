"""
cal_area.py
Overall area calculation - calculated ONCE (physical size doesn't change)
"""

import config
from perip.Driver import calculate_driver_area_single
from perip.Integrator import calculate_integrator_area_single
from perip.ADC import calculate_adc_area_single
from perip.Comparator import calculate_comparator_area_single


def calculate_overall_area():
    """
    Calculate total chip area (ONCE - doesn't depend on operations)
    
    Returns:
        dict: Area breakdown
    """
    
    # ========================
    # 1. RRAM Array Area (ONCE)
    # ========================
    
    A_array_Q = config.Q_D_IN * config.Q_D_OUT * (config.CELL_PITCH ** 2)
    A_array_K = config.K_D_IN * config.K_D_OUT * (config.CELL_PITCH ** 2)
    A_array_total = A_array_Q + A_array_K
    
    # ========================
    # 2. Q Path Peripheral Area (ONCE)
    # ========================
    
    # Single component areas
    A_driver_unit = calculate_driver_area_single(10e-6, config.TECH_NODE)
    A_integrator_unit = calculate_integrator_area_single(config.TECH_NODE)
    A_adc_unit = calculate_adc_area_single(
        config.ADC_RESOLUTION, 
        config.C_ADC_UNIT, 
        config.TECH_NODE
    )
    
    # Q path total (한 번만 계산)
    A_Q_driver = config.Q_D_IN * A_driver_unit
    A_Q_integrator = config.Q_D_OUT * A_integrator_unit
    A_Q_adc = config.Q_D_OUT * A_adc_unit
    A_Q_peripheral = A_Q_driver + A_Q_integrator + A_Q_adc
    
    # ========================
    # 3. K Path Peripheral Area (ONCE)
    # ========================
    
    A_comparator_unit = calculate_comparator_area_single(config.TECH_NODE)
    
    # K path total (한 번만 계산)
    A_K_driver = config.K_D_IN * A_driver_unit
    A_K_integrator = config.K_D_OUT * A_integrator_unit
    A_K_comparator = config.K_D_OUT * A_comparator_unit
    A_K_peripheral = A_K_driver + A_K_integrator + A_K_comparator
    
    # ========================
    # 4. Control Logic Area (ONCE)
    # ========================
    
    A_control = A_array_total * 0.1
    
    # ========================
    # Total Area (ONCE)
    # ========================
    
    A_peripheral_total = A_Q_peripheral + A_K_peripheral + A_control
    # A_total = A_array_total + A_peripheral_total
    A_total = A_array_total 
    
    return {
        'A_array_Q': A_array_Q,
        'A_array_K': A_array_K,
        'A_array_total': A_array_total,
        'A_Q_driver': A_Q_driver,
        'A_Q_integrator': A_Q_integrator,
        'A_Q_adc': A_Q_adc,
        'A_Q_peripheral': A_Q_peripheral,
        'A_K_driver': A_K_driver,
        'A_K_integrator': A_K_integrator,
        'A_K_comparator': A_K_comparator,
        'A_K_peripheral': A_K_peripheral,
        'A_control': A_control,
        'A_peripheral_total': A_peripheral_total,
        'A_total': A_total,
        # Unit areas for reference
        'A_driver_unit': A_driver_unit,
        'A_integrator_unit': A_integrator_unit,
        'A_adc_unit': A_adc_unit,
        'A_comparator_unit': A_comparator_unit
    }


def print_area_breakdown(area_result):
    """Print detailed area breakdown"""
    
    print("\n" + "="*80)
    print("Area Breakdown (Physical Size - Calculated Once)")
    print("="*80)
    print(f"{'Component':<35} {'Count':<10} {'Unit (μm²)':<15} {'Total (μm²)':<15}")
    print("-"*80)
    
    # Array
    print(f"{'RRAM Arrays:':<35}")
    print(f"{'  Q Array (15×30)':<35} {'450':<10} {'9.00':<15} {area_result['A_array_Q']*1e12:<15.2f}")
    print(f"{'  K Array (19×30)':<35} {'570':<10} {'9.00':<15} {area_result['A_array_K']*1e12:<15.2f}")
    
    print()
    
    # Q Path
    print(f"{'Q Path Peripherals:':<35}")
    print(f"{'  Drivers':<35} {config.Q_D_IN:<10} {area_result['A_driver_unit']*1e12:<15.2f} {area_result['A_Q_driver']*1e12:<15.2f}")
    print(f"{'  Integrators':<35} {config.Q_D_OUT:<10} {area_result['A_integrator_unit']*1e12:<15.2f} {area_result['A_Q_integrator']*1e12:<15.2f}")
    print(f"{'  ADCs (7-bit)':<35} {config.Q_D_OUT:<10} {area_result['A_adc_unit']*1e12:<15.2f} {area_result['A_Q_adc']*1e12:<15.2f}")
    
    print()
    
    # K Path
    print(f"{'K Path Peripherals:':<35}")
    print(f"{'  Drivers':<35} {config.K_D_IN:<10} {area_result['A_driver_unit']*1e12:<15.2f} {area_result['A_K_driver']*1e12:<15.2f}")
    print(f"{'  Integrators':<35} {config.K_D_OUT:<10} {area_result['A_integrator_unit']*1e12:<15.2f} {area_result['A_K_integrator']*1e12:<15.2f}")
    print(f"{'  Comparators':<35} {config.K_D_OUT:<10} {area_result['A_comparator_unit']*1e12:<15.2f} {area_result['A_K_comparator']*1e12:<15.2f}")
    
    print()
    print(f"{'Control Logic (10% of array)':<35} {'-':<10} {'-':<15} {area_result['A_control']*1e12:<15.2f}")
    
    print("-"*80)
    print(f"{'TOTAL CHIP AREA':<35} {'':<10} {'':<15} {area_result['A_total']*1e12:<15.2f}")
    print(f"{'(in mm²)':<35} {'':<10} {'':<15} {area_result['A_total']*1e6:<15.6f}")
    print("="*80 + "\n")


if __name__ == "__main__":
    print("Testing cal_area.py")
    
    area_result = calculate_overall_area()
    print_area_breakdown(area_result)
    
    print(f"\nTotal chip area: {area_result['A_total']*1e6:.6f} mm²")