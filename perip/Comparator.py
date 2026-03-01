"""
Comparator.py
Comparator energy calculation
"""

def calculate_comparator_energy(C_comp, V_supply):
    """
    Calculate comparator dynamic energy
    
    E = 0.5 × C × V² × N
    
    Args:
        C_comp (float): Comparator input capacitance (F)
        V_supply (float): Supply voltage (V)
        num_operations (int): Number of comparisons
    
    Returns:
        float: Total dynamic energy (J)
    """
    return 0.5 * C_comp * (V_supply ** 2) 


def calculate_comparator_leakage_energy(P_leak, time, num_comparators):
    """
    Calculate comparator leakage energy
    
    Args:
        P_leak (float): Leakage power per comparator (W)
        time (float): Total time (s)
        num_comparators (int): Number of comparators
    
    Returns:
        float: Total leakage energy (J)
    """
    return P_leak * time * num_comparators

def calculate_comparator_area_single(tech_node):
    """
    Calculate area of a SINGLE comparator
    
    Args:
        tech_node (float): Technology node (m)
    
    Returns:
        float: Area of one comparator (m²)
    """
    transistor_width = 2e-6
    gate_length = tech_node
    layout_factor = 10
    num_transistors = 10
    
    area = transistor_width * gate_length * layout_factor * num_transistors
    
    return area

if __name__ == "__main__":
    print("Testing Comparator.py")
    print("="*60)
    
    import config
    
    E_comp = calculate_comparator_energy(
        config.C_COMPARATOR_IN, 
        config.V_SUPPLY, 
        1000
    )
    print(f"Comparator energy (1000 ops): {E_comp*1e12:.3f} pJ")
    
    E_leak = calculate_comparator_leakage_energy(
        config.P_LEAK_COMPARATOR, 
        1e-3, 
        30
    )
    print(f"Comparator leakage (30 units, 1ms): {E_leak*1e6:.3f} μJ")

    A_single = calculate_comparator_area_single(config.TECH_NODE)
    print(f"Area of single comparator: {A_single*1e12:.3f} μm²")