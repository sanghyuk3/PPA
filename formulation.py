import math

# RC 회로의 신호 전파 지연을 계산하는 함수
def horowitz(tr, beta, ramp_input):
    # Edge cases
    if tr <= 0 or ramp_input <= 0:
        return 0, ramp_input
    
    alpha = 1.0 / (ramp_input * tr)
    vs = 0.5  
    beta = 0.5  

    try:
        delay = tr * math.sqrt(math.log(vs)**2 + 2*alpha*beta*(1-vs))
    except (ValueError, OverflowError):
        delay = tr  

    ramp_output = (1 - vs) / delay if delay > 0 else ramp_input
    
    return delay, ramp_output

# RC stage를 거치는 동안 총 지연 계산
def calculate_chain_delay(RC_list, initial_ramp=1e9):
    total_delay = 0
    ramp = initial_ramp
    
    for R, C in RC_list:
        tr = R * C
        delay, ramp = horowitz(tr, 0.5, ramp)
        total_delay += delay
    
    return total_delay, ramp

# MOSFET이 ON일 때의 저항 계산
def calculate_on_resistance(width, temperature=300, tech_node=65e-9):
    base_resistance = 1000  # Ohms for reference width
    reference_width = 1e-6  # 1μm
    
    resistance = base_resistance * (reference_width / width)
    
    # Temperature dependency (rough approximation)
    temp_factor = temperature / 300
    resistance *= temp_factor
    
    return resistance

# 금속 배선의 저항
def calculate_wire_resistance(length, width, tech_node=65e-9):
    resistivity = 1.7e-8  # Copper
    thickness = tech_node  # Assume thickness = tech node
    
    area = width * thickness
    resistance = resistivity * length / area if area > 0 else 0
    
    return resistance

# MOSFET의 gate CAP
def calculate_gate_cap(width, tech_node=65e-9):
    cox_per_area = 10e-15 / (1e-6)**2  # F/m²
    
    length = tech_node
    capacitance = cox_per_area * width * length
    
    return capacitance

# 금속 배선의 기생 CAP
def calculate_wire_cap(length, width, tech_node=65e-9):
    cap_per_length = 0.2e-15 / 1e-6  # F/m
    
    capacitance = cap_per_length * length
    
    return capacitance

# switching energy
def calculate_dynamic_energy(capacitance, voltage, num_transitions):
    return 0.5 * capacitance * (voltage ** 2) * num_transitions

# 대기 전력 소모
def calculate_leakage_energy(leakage_power, time):
    return leakage_power * time

# 저항에서의 열 손실
def calculate_resistive_energy(resistance, current, time):
    return (current ** 2) * resistance * time


