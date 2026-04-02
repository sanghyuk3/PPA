"""
conventional_rram.py
Conventional RRAM (KT thermally-switched) PPA Calculator

KT write 기반 일반 RRAM:
  - SET/RESET 시 Joule heating으로 filament 형성/파괴 (thermal activation)
  - Write 전압/전류가 높고 pulse 길어 write energy가 큼
  - Read settling 시간도 ISAAC 이상적 값보다 길다

ISAAC IMC와의 핵심 차이:
  - ISAAC: weight를 RRAM에 한 번 기록 후 inference는 analog MAC (write 제외)
  - Conventional: write energy가 매우 크고, read도 느려 inference 레이턴시 증가
"""

import config
from Cell import calculate_rram_read_energy, calculate_rram_write_energy
from perip.perip import calculate_energy_perip
from formulation import calculate_chain_delay
from PPA.cal_area import calculate_overall_area

# ============================================================
# Conventional RRAM Parameters (KT 열적 스위칭, HfOx/TaOx 계열)
# 참고: ISSCC/IEDM 논문 및 RRAM 리뷰 논문 기반
# ============================================================
CONV_V_WRITE_SET    = 2.0      # SET 전압 (V) - filament 형성
CONV_I_COMPLIANCE   = 100e-6   # 전류 compliance (A) - Joule heating 제한
CONV_T_WRITE_SET    = 500e-9   # SET pulse 폭 (s)

CONV_V_WRITE_RESET  = 1.5      # RESET 전압 (V) - filament 파괴
CONV_I_RESET        = 200e-6   # RESET peak 전류 (A) - 더 큰 전류로 빠른 파괴
CONV_T_WRITE_RESET  = 200e-9   # RESET pulse 폭 (s)

# Read: ISAAC 이상적(10ns) 대비 현실적인 sense-amp settling 시간
CONV_T_READ         = 120e-9   # 120ns (문헌상 conventional RRAM 읽기 시간)

# G_ON/G_OFF는 ISAAC과 동일 (같은 RRAM 물질 가정, 차이는 write 방식)
CONV_G_ON  = config.G_ON
CONV_G_OFF = config.G_OFF


def calculate_conventional_rram_ppa():
    """
    Conventional RRAM 전체 PPA 계산.

    Returns:
        dict: runtime, energy, area, TOPS, TOPS_per_W, TOPS_per_mm2,
              energy_per_sample, mac_ops, E_write, E_inference, T_write, T_inference
    """

    # ========================
    # MAC 연산 수 (ISAAC과 동일 workload)
    # ========================
    total_q_macs = (config.RRAM_NUM_SAMPLES * config.RRAM_LAYERS *
                    config.Q_READ_MULTIPLIER * config.Q_D_IN * config.Q_D_OUT)
    total_k_macs = (config.RRAM_NUM_SAMPLES * config.RRAM_LAYERS *
                    config.K_READ_MULTIPLIER * config.K_D_IN * config.K_D_OUT)
    total_mac_ops = total_q_macs + total_k_macs

    # ========================
    # 1. Write Phase (KT thermal switching)
    #    weight programming: 배열에 한 번 기록
    # ========================
    E_set_per_cell   = calculate_rram_write_energy(CONV_V_WRITE_SET,   CONV_I_COMPLIANCE, CONV_T_WRITE_SET)
    E_reset_per_cell = calculate_rram_write_energy(CONV_V_WRITE_RESET, CONV_I_RESET,      CONV_T_WRITE_RESET)
    # 평균: ternary weight → SET:RESET 비율 ≈ 1:1
    E_write_per_cell = (E_set_per_cell + E_reset_per_cell) / 2.0

    # 전체 weight cell 수 (Q + K, 12 layer)
    total_Q_cells = config.Q_D_IN  * config.Q_D_OUT * config.RRAM_LAYERS
    total_K_cells = config.K_D_IN  * config.K_D_OUT * config.RRAM_LAYERS
    total_cells   = total_Q_cells + total_K_cells

    E_write_total = total_cells * E_write_per_cell

    # Write latency: row-parallel (같은 행의 모든 열을 동시에 write)
    T_write_per_row = CONV_T_WRITE_SET + CONV_T_WRITE_RESET       # SET + RESET per row
    T_write_Q = config.Q_D_IN * config.RRAM_LAYERS * T_write_per_row
    T_write_K = config.K_D_IN * config.RRAM_LAYERS * T_write_per_row
    T_write_total = T_write_Q + T_write_K

    # ========================
    # 2. Read Phase (Inference) - 느린 T_READ 적용
    # ========================
    E_read_cell = calculate_rram_read_energy(
        config.V_READ, CONV_G_ON, CONV_G_OFF, CONV_T_READ, sparsity=0.05
    )
    E_read_total = total_mac_ops * E_read_cell

    # RC delay chain (ISAAC과 동일 회로, T_READ만 교체)
    RC_list_Q = [
        (config.R_DRIVER,       config.C_WORDLINE),
        (config.R_BITLINE_WIRE, config.C_INTEGRATOR_IN),
        (config.R_INTEGRATOR,   config.C_ADC_IN),
    ]
    delay_Q_chain, _ = calculate_chain_delay(RC_list_Q, config.INITIAL_RAMP)
    delay_Q_per_op = (delay_Q_chain + CONV_T_READ +
                      config.T_DRIVER + config.T_INTEGRATOR + config.T_ADC)

    RC_list_K = [
        (config.R_DRIVER,       config.C_WORDLINE),
        (config.R_BITLINE_WIRE, config.C_INTEGRATOR_IN),
        (config.R_INTEGRATOR,   config.C_COMPARATOR_IN),
    ]
    delay_K_chain, _ = calculate_chain_delay(RC_list_K, config.INITIAL_RAMP)
    delay_K_per_op = (delay_K_chain + CONV_T_READ +
                      config.T_DRIVER + config.T_INTEGRATOR + config.T_COMPARATOR)

    total_q_ops = (config.RRAM_NUM_SAMPLES * config.RRAM_LAYERS *
                   config.Q_READ_MULTIPLIER)
    total_k_ops = (config.RRAM_NUM_SAMPLES * config.RRAM_LAYERS *
                   config.K_READ_MULTIPLIER)

    T_read_Q = delay_Q_per_op * total_q_ops
    T_read_K = delay_K_per_op * total_k_ops
    T_read_total = T_read_Q + T_read_K

    # Peripheral energy (leakage 비중 큼 – read 시간이 길어서)
    num_Q_input  = total_q_macs // config.Q_D_OUT
    num_Q_output = total_q_macs // config.Q_D_IN
    num_K_input  = total_k_macs // config.K_D_OUT
    num_K_output = total_k_macs // config.K_D_IN

    result_Q = calculate_energy_perip(num_Q_input, num_Q_output, T_read_total, 'Q')
    result_K = calculate_energy_perip(num_K_input, num_K_output, T_read_total, 'K')
    E_perip_total = result_Q['total'] + result_K['total']

    E_inference = E_read_total + E_perip_total

    # ========================
    # 3. 합산
    # ========================
    E_total  = E_write_total + E_inference
    T_total  = T_write_total + T_read_total

    area_result = calculate_overall_area()
    area_mm2    = area_result['A_total'] * 1e6   # m² → mm²

    power        = E_total / T_total if T_total > 0 else 0
    tops         = (total_mac_ops * 2) / T_total / 1e12 if T_total > 0 else 0
    tops_per_w   = tops / power if power > 0 else 0
    tops_per_mm2 = tops / area_mm2 if area_mm2 > 0 else 0
    energy_per_sample = E_total / config.RRAM_NUM_SAMPLES

    return {
        'runtime':          T_total,
        'T_write':          T_write_total,
        'T_inference':      T_read_total,
        'energy':           E_total,
        'E_write':          E_write_total,
        'E_inference':      E_inference,
        'power':            power,
        'area':             area_mm2,
        'mac_ops':          total_mac_ops,
        'TOPS':             tops,
        'TOPS_per_W':       tops_per_w,
        'TOPS_per_mm2':     tops_per_mm2,
        'energy_per_sample': energy_per_sample,
        # 셀 당 write energy (정보용)
        'E_write_per_cell_pJ': E_write_per_cell * 1e12,
    }


def print_conventional_rram_summary(r):
    """계산 결과 출력"""
    print("\n" + "="*80)
    print("Conventional RRAM (KT Write) PPA Summary")
    print("="*80)
    print(f"  Write parameters :")
    print(f"    SET  : {CONV_V_WRITE_SET}V, {CONV_I_COMPLIANCE*1e6:.0f}μA, {CONV_T_WRITE_SET*1e9:.0f}ns")
    print(f"    RESET: {CONV_V_WRITE_RESET}V, {CONV_I_RESET*1e6:.0f}μA, {CONV_T_WRITE_RESET*1e9:.0f}ns")
    print(f"    Avg E/cell : {r['E_write_per_cell_pJ']:.1f} pJ")
    print(f"  Read  : {CONV_T_READ*1e9:.0f}ns (vs ISAAC ideal {config.T_READ*1e9:.0f}ns)")
    print("-"*80)
    print(f"  Write latency   : {r['T_write']*1e3:.3f} ms")
    print(f"  Inference latency: {r['T_inference']*1e3:.3f} ms")
    print(f"  Total runtime   : {r['runtime']*1e3:.3f} ms")
    print(f"  Write energy    : {r['E_write']*1e3:.4f} mJ")
    print(f"  Inference energy: {r['E_inference']*1e3:.4f} mJ")
    print(f"  Total energy    : {r['energy']*1e3:.4f} mJ")
    print(f"  Power           : {r['power']*1e3:.4f} mW")
    print(f"  Area            : {r['area']:.6f} mm²")
    print(f"  TOPS            : {r['TOPS']:.6f}")
    print(f"  TOPS/W          : {r['TOPS_per_W']:.4f}")
    print(f"  TOPS/mm²        : {r['TOPS_per_mm2']:.6f}")
    print(f"  Energy/sample   : {r['energy_per_sample']*1e6:.3f} μJ")
    print("="*80 + "\n")


if __name__ == "__main__":
    result = calculate_conventional_rram_ppa()
    print_conventional_rram_summary(result)
