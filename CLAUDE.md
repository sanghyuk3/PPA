# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an RRAM-based (Resistive Random Access Memory) transformer inference simulator with detailed PPA (Power, Performance, Area) analysis. It models the ISAAC architecture for in-memory computing, comparing it against GPU baselines (V100, A100, H100, RTX3090, RTX4090). The workload is ALBERT on SST-2 classification with ternary quantization.

## Running the Project

```bash
# Run the full pipeline (quantize model, compute RRAM PPA, evaluate accuracy, compare vs GPUs)
python main.py

# Test individual modules (each has its own __main__ block)
python config.py              # Print all hardware configuration parameters
python param.py               # Test the top-level ISAAC_RRAM_PPA calculator
python Cell.py                # Test RRAM cell read/write energy
python formulation.py         # Test Horowitz RC delay formulas
python PPA/cal_latency.py     # Test latency calculation
python PPA/cal_energy.py      # Test energy calculation
python PPA/cal_area.py        # Test area calculation
python perip/Driver.py        # Test driver energy and area
python perip/Integrator.py    # Test integrator energy and area
python perip/ADC.py           # Test ADC energy and area
python perip/Comparator.py    # Test comparator energy and area
python GPU/GPU.py             # Test GPU PPA baselines
```

**Dependencies** (no requirements.txt): `torch`, `transformers`, `datasets`

## Architecture

### Data Flow

```
config.py (65 hardware params) → param.py (ISAAC_RRAM_PPA)
                                       ├── PPA/cal_latency.py  ← formulation.py (Horowitz)
                                       ├── PPA/cal_energy.py   ← Cell.py, perip/perip.py
                                       └── PPA/cal_area.py     ← perip/{Driver,Integrator,ADC,Comparator}

main.py
  ├── Inference.py       (ternary quantization of ALBERT Q/K/V projections)
  ├── param.py           (RRAM PPA)
  ├── evaluation.py      (accuracy + derived metrics: TOPS, TOPS/W, energy/sample)
  └── GPU/GPU.py         (GPU baselines for comparison table)
```

### Key Architectural Concepts

**Dual-Path RRAM Architecture**: Q and K attention projections follow separate analog paths:
- **Q path**: Driver → RRAM array → Integrator → 7-bit SAR ADC
- **K path**: Driver → RRAM array → Integrator → 1-bit Comparator

**RC Delay Chain** (`formulation.py`): Uses the Horowitz approximation to compute realistic analog propagation delays through cascaded driver/wire/cell/sense-amp stages. `calculate_chain_delay()` takes a list of `(R, C)` stage pairs.

**Ternary Quantization** (`Inference.py`): Weights and activations quantized to `{-0.5, 0, +0.5}`. `TernaryQuantizeFn` is a custom `torch.autograd.Function`. `apply_quantlinear_with_stats()` replaces only Q/K/V projections in ALBERT with `QuantLinearWithStats`.

**PPA Aggregation** (`param.py`): `ISAAC_RRAM_PPA` calls the three `cal_*` modules and returns a dict with latency (s), energy (J), area (mm²), TOPS, TOPS/W, and energy/sample.

### Configuration (`config.py`)

All hardware parameters live here as module-level globals — no class or dataclass. Key values:
- Technology: 65nm, cell pitch 30μm
- Read voltage: 0.2V, G_ON=6.5μS, G_OFF=1.0μS, t_read=120μs
- Array dimensions: Q 256×256, K 256×256, 12 layers, 36 tokens
- ADC: 7-bit SAR

When changing hardware parameters, only edit `config.py`; all downstream modules import from it.

### GPU Baselines (`GPU/GPU.py`)

`GPU_PPA` class holds specs (TOPS, TDP, die area) for each GPU. Computes runtime and energy for the QKᵀ operation given the same workload as RRAM.
