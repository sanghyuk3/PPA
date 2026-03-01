"""
ADC.py
ADC energy calculation (SAR ADC model)
"""
import config

def calculate_adc_energy(resolution, V_ref, C_unit):
    """
    Calculate ADC conversion energy (SAR ADC)
    
    E = 2^N × C × V²
    
    Args:
        resolution (int): ADC resolution (bits)
        V_ref (float): Reference voltage (V)
        C_unit (float): Unit capacitance (F)
    
    Returns:
        float: Energy per conversion (J)
    """
    num_comparisons = 2 ** resolution
    energy = num_comparisons * C_unit * (V_ref ** 2)
    return energy


def calculate_adc_leakage_energy(P_leak, time, num_adcs):
    """
    Calculate ADC leakage energy
    
    Args:
        P_leak (float): Leakage power per ADC (W)
        time (float): Total time (s)
        num_adcs (int): Number of ADCs
    
    Returns:
        float: Total leakage energy (J)
    """
    return P_leak * time * num_adcs

def calculate_adc_area_single(resolution, C_unit, tech_node):
    """
    Calculate area of a SINGLE ADC
    
    Args:
        resolution (int): ADC resolution (bits)
        C_unit (float): Unit capacitance (F)
        tech_node (float): Technology node (m)
    
    Returns:
        float: Area of one ADC (m²)
    """
    # 1. DAC area (capacitor array)
    num_caps = 2 ** resolution
    cox_per_area = 10e-15 / (1e-6)**2  # 10 fF/μm²
    cap_area = C_unit / cox_per_area
    spacing_factor = 2
    dac_area = num_caps * cap_area * spacing_factor
    
    # 2. Comparator area
    comparator_area = 50e-12  # 50 μm²
    
    # 3. SAR Logic area
    logic_area = resolution * 10e-12  # ~10 μm² per bit
    
    # Total for ONE ADC
    total_area = dac_area + comparator_area + logic_area
    
    return total_area


if __name__ == "__main__":
    print("Testing ADC.py")
    print("="*60)
    
    E_adc = calculate_adc_energy(
        config.ADC_RESOLUTION, 
        config.V_SUPPLY, 
        config.C_ADC_UNIT
    )
    print(f"7-bit ADC energy per conversion: {E_adc*1e15:.3f} fJ")
    
    E_leak = calculate_adc_leakage_energy(
        config.P_LEAK_ADC, 
        1e-3, 
        30
    )
    print(f"ADC leakage (30 units, 1ms): {E_leak*1e6:.3f} μJ")

    A_single = calculate_adc_area_single(
        config.ADC_RESOLUTION, 
        config.C_ADC_UNIT, 
        config.TECH_NODE
    )
    print(f"Area of single ADC: {A_single*1e12:.3f} μm²")
    
    # Example: 30 ADCs
    num_ADCs = 30
    total_area = num_ADCs * A_single
    print(f"\nTotal area for {num_ADCs} ADCs: {total_area*1e12:.3f} μm²")