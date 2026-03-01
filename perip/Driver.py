"""
Driver.py
Driver circuit energy calculation
"""

def calculate_driver_energy(V_supply, C_load):
    """
    Calculate driver energy (charging capacitive load)
    
    E = C_load × V²
    
    Args:
        V_supply (float): Supply voltage (V)
        C_load (float): Load capacitance (F)
    
    Returns:
        float: Energy per activation (J)
    """
    return C_load * (V_supply ** 2)


def calculate_driver_leakage_energy(P_leak, time, num_drivers):
    """
    Calculate driver leakage energy
    
    Args:
        P_leak (float): Leakage power per driver (W)
        time (float): Total time (s)
        num_drivers (int): Number of drivers
    
    Returns:
        float: Total leakage energy (J)
    """
    return P_leak * time * num_drivers

def calculate_driver_area_single(width_nmos, tech_node):
    """
    Calculate area of a SINGLE driver
    
    Args:
        width_nmos (float): NMOS width (m)
        tech_node (float): Technology node (m)
    
    Returns:
        float: Area of one driver (m²)
    """
    gate_length = tech_node
    layout_factor = 15
    num_stages = 3
    width_pmos = width_nmos * 2
    
    area_per_stage = (width_nmos + width_pmos) * gate_length * layout_factor
    total_area = area_per_stage * num_stages
    
    return total_area

if __name__ == "__main__":
    print("Testing Driver.py")
    print("="*60)
    
    import config
    
    E_driver = calculate_driver_energy(config.V_SUPPLY, config.C_DRIVER_OUTPUT)
    print(f"Driver energy per activation: {E_driver*1e15:.3f} fJ")
    
    E_leak = calculate_driver_leakage_energy(config.P_LEAK_DRIVER, 1e-3, 15)
    print(f"Driver leakage (15 drivers, 1ms): {E_leak*1e6:.3f} μJ")
    
    A_single = calculate_driver_area_single(10e-6, config.TECH_NODE)
    print(f"Area of single driver: {A_single*1e12:.3f} μm²")