"""
Integrator.py
Charge integrator energy calculation
"""

def calculate_integrator_energy(C_integrator, V_supply):
    """
    Calculate integrator dynamic energy
    
    E = 0.5 × C × V² × N
    
    Args:
        C_integrator (float): Integrator capacitance (F)
        V_supply (float): Supply voltage (V)
        num_operations (int): Number of operations
    
    Returns:
        float: Total dynamic energy (J)
    """
    return 0.5 * C_integrator * (V_supply ** 2) 


def calculate_integrator_leakage_energy(P_leak, time, num_integrators):
    """
    Calculate integrator leakage energy
    
    Args:
        P_leak (float): Leakage power per integrator (W)
        time (float): Total time (s)
        num_integrators (int): Number of integrators
    
    Returns:
        float: Total leakage energy (J)
    """
    return P_leak * time * num_integrators

def calculate_integrator_area_single(tech_node):
    """
    Calculate area of a SINGLE integrator (OPA)
    
    Args:
        tech_node (float): Technology node (m)
    
    Returns:
        float: Area of one integrator (m²)
    """
    avg_transistor_width = 5e-6
    gate_length = tech_node
    layout_factor = 20
    num_transistors = 20
    
    area_per_transistor = avg_transistor_width * gate_length * layout_factor
    total_area = area_per_transistor * num_transistors
    
    return total_area

if __name__ == "__main__":
    print("Testing Integrator.py")
    print("="*60)
    
    import config
    
    E_int = calculate_integrator_energy(
        config.C_INTEGRATOR_IN, 
        config.V_SUPPLY, 
        1000
    )
    print(f"Integrator energy (1000 ops): {E_int*1e12:.3f} pJ")
    
    E_leak = calculate_integrator_leakage_energy(
        config.P_LEAK_INTEGRATOR, 
        1e-3, 
        30
    )
    print(f"Integrator leakage (30 units, 1ms): {E_leak*1e6:.3f} μJ")

    A_single = calculate_integrator_area_single(config.TECH_NODE)
    print(f"Area of single integrator: {A_single*1e12:.3f} μm²")