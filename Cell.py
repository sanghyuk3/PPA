"""
Cell.py
RRAM Cell energy calculation
"""

def calculate_rram_read_energy_gavg(V_read, G_avg, t_read):
    """
    Calculate RRAM cell read energy using directly measured average conductance.
    Use this instead of calculate_rram_read_energy when G_avg is measured.

    Args:
        V_read (float): Read voltage (V)
        G_avg (float): Measured average conductance (S)
        t_read (float): Read time (s)

    Returns:
        float: Energy per cell read (J)
    """
    I_avg = G_avg * V_read
    return V_read * I_avg * t_read


def calculate_rram_read_energy(V_read, G_on, G_off, t_read, sparsity=0.05):
    """
    Calculate RRAM cell read energy with sparsity consideration
    
    Args:
        V_read (float): Read voltage (V)
        G_on (float): ON conductance (S)
        G_off (float): OFF conductance (S)
        t_read (float): Read time (s)
        sparsity (float): Fraction of ON cells (0~1)
    
    Returns:
        float: Energy per cell read (J)
    """
    I_avg = (sparsity * G_on + (1 - sparsity) * G_off) * V_read
    energy = V_read * I_avg * t_read
    return energy


def calculate_rram_write_energy(V_write, I_write, t_write):
    """
    Calculate RRAM cell write energy
    
    Args:
        V_write (float): Write voltage (V)
        I_write (float): Write current (A)
        t_write (float): Write time (s)
    
    Returns:
        float: Energy per cell write (J)
    """
    energy = V_write * I_write * t_write
    return energy


if __name__ == "__main__":
    print("Testing Cell.py")
    print("="*60)
    
    # Test RRAM read energy
    V_read = 0.2
    G_on = 6.5e-6
    G_off = 1.0e-6
    t_read = 3e-9
    
    E_read = calculate_rram_read_energy(V_read, G_on, G_off, t_read, 0.5)
    print(f"RRAM read energy: {E_read*1e15:.3f} fJ")
    
    # Test RRAM write energy
    E_write = calculate_rram_write_energy(1.0, 100e-6, 100e-9)
    print(f"RRAM write energy: {E_write*1e12:.3f} pJ")