"""
glue_eval.py
BERT RRAM inference accuracy across multiple GLUE tasks.

모든 task에 동일하게 적용:
  - W4A8 inference quantization (Inference.py의 quantize_weight_4bit / quantize_activation_8bit)
  - 동일 RRAM hardware variation (config.VARIATION_* 통계, 1회 고정 샘플링)
  - bias = None (SST-2 학습 조건과 통일)
  - GPU 사용 (Colab 기준)
"""
import torch
import random
import numpy as np
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import BertTokenizer, AutoModelForSequenceClassification

import config
from Inference import apply_quantlinear_with_stats

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# 재현성 보장
# ============================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)
MAX_LEN = 128

# ============================================================
# GLUE Task Configs
# ============================================================
GLUE_TASKS = {
    'sst2': {
        'dataset': ('glue', 'sst2'),
        'split':   'validation',
        'keys':    ('sentence', None),
        'num_labels': 2,
        'model':   'textattack/bert-base-uncased-SST-2',
    },
    'mrpc': {
        'dataset': ('glue', 'mrpc'),
        'split':   'validation',
        'keys':    ('sentence1', 'sentence2'),
        'num_labels': 2,
        'model':   'textattack/bert-base-uncased-MRPC',
    },
    'mnli': {
        'dataset': ('glue', 'mnli'),
        'split':   'validation_matched',
        'keys':    ('premise', 'hypothesis'),
        'num_labels': 3,
        'model':   'textattack/bert-base-uncased-MNLI',
    },
}


def _make_loader(task_name, tokenizer, batch_size=128):
    cfg = GLUE_TASKS[task_name]
    k1, k2 = cfg['keys']
    data = load_dataset(*cfg['dataset'])[cfg['split']]

    def encode(batch):
        if k2 is None:
            return tokenizer(batch[k1], truncation=True,
                             padding='max_length', max_length=MAX_LEN)
        return tokenizer(batch[k1], batch[k2], truncation=True,
                         padding='max_length', max_length=MAX_LEN)

    enc = data.map(encode, batched=True)
    enc.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

    def collate(batch):
        return {
            'input_ids':      torch.stack([x['input_ids']      for x in batch]),
            'attention_mask': torch.stack([x['attention_mask'] for x in batch]),
            'labels':         torch.tensor([x['label']         for x in batch]),
        }

    return DataLoader(enc, batch_size=batch_size, shuffle=False,
                      collate_fn=collate), len(data)


def evaluate_task(task_name, local_sst2_ckpt=None, local_ckpts=None):
    """
    하나의 GLUE task 평가.
    모든 task에 W4A8 quantization + RRAM variation 동일 적용.
    local_ckpts: dict {task_name: ckpt_path} for W4A8 QAT checkpoints
    Returns (accuracy, num_samples).
    """
    import os
    cfg = GLUE_TASKS[task_name]
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # W4A8 QAT 로컬 checkpoint 사용
    # SST-2: local_sst2_ckpt 필수 (없으면 에러)
    # MRPC/MNLI 등: local_ckpts에서 찾음, 없으면 textattack FP32 fallback
    from transformers import BertForSequenceClassification
    local_ckpt = None
    if task_name == 'sst2':
        if not local_sst2_ckpt or not os.path.exists(local_sst2_ckpt):
            raise FileNotFoundError(
                f"SST-2 W4A8 checkpoint not found: {local_sst2_ckpt}\n"
                "Upload W4A8_BERT_best_acc0.9174.pt to /content/ on Colab.")
        local_ckpt = local_sst2_ckpt
    elif local_ckpts and task_name in local_ckpts and os.path.exists(local_ckpts[task_name]):
        local_ckpt = local_ckpts[task_name]

    if local_ckpt:
        model = BertForSequenceClassification.from_pretrained(
            'bert-base-uncased', num_labels=cfg['num_labels'])
        sd = torch.load(local_ckpt, map_location='cpu')
        if isinstance(sd, dict) and 'model_state_dict' in sd:
            sd = sd['model_state_dict']
        model.load_state_dict(sd, strict=False)
        force_no_bias = True  # W4A8 QAT 학습 시 bias=None
    else:
        model = AutoModelForSequenceClassification.from_pretrained(cfg['model'])
        force_no_bias = False  # textattack FP32 모델은 bias 유지

    apply_quantlinear_with_stats(model, force_no_bias=force_no_bias)
    model.to(DEVICE)
    model.eval()

    loader, n_samples = _make_loader(task_name, tokenizer)

    correct, total = 0, 0
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            out = model(**batch)
            preds = out.logits.argmax(dim=-1)
            correct += (preds == batch['labels']).sum().item()
            total += batch['labels'].size(0)

    return correct / total, n_samples


def run_all_glue(local_sst2_ckpt=None, local_ckpts=None):
    """전체 GLUE task 실행. Returns dict: task → {accuracy, num_samples}.
    local_ckpts: dict {task_name: ckpt_path} for W4A8 QAT checkpoints
    """
    print(f"Device: {DEVICE}")
    results = {}
    for task in GLUE_TASKS:
        print(f"  [{task.upper():5s}] evaluating...", end=' ', flush=True)
        try:
            acc, n = evaluate_task(task, local_sst2_ckpt=local_sst2_ckpt,
                                   local_ckpts=local_ckpts)
            results[task] = {'accuracy': acc, 'num_samples': n}
            print(f"acc={acc*100:.2f}%  (n={n})")
        except Exception as e:
            print(f"FAILED: {e}")
            results[task] = {'accuracy': None, 'num_samples': 0}
    return results
