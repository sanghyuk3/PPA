import config
import torch

def evaluate_with_ppa(model, loader, rram_results):
    device = torch.device("cpu")

    model.eval()
    correct, total = 0, 0
    
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            preds = outputs.logits.argmax(dim=-1)
            correct += (preds == batch["labels"]).sum().item()
            total += batch["labels"].size(0)

    # ============================
    # total mac
    # ============================ 
    total_q_macs = config.RRAM_NUM_SAMPLES * config.RRAM_LAYERS * config.Q_READ_MULTIPLIER * config.Q_D_IN * config.Q_D_OUT 
    total_k_macs = config.RRAM_NUM_SAMPLES * config.RRAM_LAYERS * config.K_READ_MULTIPLIER * config.K_D_IN * config.K_D_OUT
    total_mac_ops = total_q_macs + total_k_macs 

    # ============================
    # total PPA
    # ============================ 
    total_runtime = rram_results['runtime']
    total_energy = rram_results['energy']
    total_area = rram_results['area']

    # ============================
    # 지표 계산
    # ============================ 
    accuracy = correct / total
    power = total_energy / total_runtime 
    tops = total_mac_ops / total_runtime / 1e12 
    tops_per_w = tops / power if power > 0 else 0
    area_mm2 = total_area * 1e6
    tops_per_mm2 = tops / area_mm2 if area_mm2 > 0 else 0
    energy_per_sentence = total_energy / config.RRAM_NUM_SAMPLES
    area_per_sentence = area_mm2 / config.RRAM_NUM_SAMPLES
    
    return {
        'accuracy': accuracy,
        'runtime': total_runtime,
        'power': power,
        'energy': total_energy,
        'area': area_mm2,
        'mac_ops': total_mac_ops,
        'TOPS': tops,
        'TOPS_per_W': tops_per_w,
        'TOPS_per_mm2': tops_per_mm2,
        'energy_per_sentence' : energy_per_sentence,
        'area_per_sentence' : area_per_sentence
    }