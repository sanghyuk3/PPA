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
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import BertTokenizer, AutoModelForSequenceClassification

import config
from Inference import apply_quantlinear_with_stats

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    'cola': {
        'dataset': ('glue', 'cola'),
        'split':   'validation',
        'keys':    ('sentence', None),
        'num_labels': 2,
        'model':   'textattack/bert-base-uncased-CoLA',
    },
    'qnli': {
        'dataset': ('glue', 'qnli'),
        'split':   'validation',
        'keys':    ('question', 'sentence'),
        'num_labels': 2,
        'model':   'textattack/bert-base-uncased-QNLI',
    },
    'rte': {
        'dataset': ('glue', 'rte'),
        'split':   'validation',
        'keys':    ('sentence1', 'sentence2'),
        'num_labels': 2,
        'model':   'textattack/bert-base-uncased-RTE',
    },
    'mnli': {
        'dataset': ('glue', 'mnli'),
        'split':   'validation_matched',
        'keys':    ('premise', 'hypothesis'),
        'num_labels': 3,
        'model':   'textattack/bert-base-uncased-MNLI',
    },
    'qqp': {
        'dataset': ('glue', 'qqp'),
        'split':   'validation',
        'keys':    ('question1', 'question2'),
        'num_labels': 2,
        'model':   'textattack/bert-base-uncased-QQP',
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


def evaluate_task(task_name, local_sst2_ckpt=None):
    """
    하나의 GLUE task 평가.
    모든 task에 W4A8 quantization + RRAM variation 동일 적용.
    Returns (accuracy, num_samples).
    """
    import os
    cfg = GLUE_TASKS[task_name]
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # SST-2: 로컬 체크포인트 우선 / 나머지: HuggingFace hub
    if local_sst2_ckpt and task_name == 'sst2' and os.path.exists(local_sst2_ckpt):
        from transformers import BertForSequenceClassification
        model = BertForSequenceClassification.from_pretrained(
            'bert-base-uncased', num_labels=2)
        ckpt = torch.load(local_sst2_ckpt, map_location='cpu')
        sd = (ckpt.get('model_state_dict', ckpt.get('state_dict', ckpt))
              if isinstance(ckpt, dict) else ckpt)
        model.load_state_dict(sd, strict=False)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(cfg['model'])

    # W4A8 quantization + RRAM variation (Q/K) 적용
    # SST-2 W4A8 checkpoint: bias=None으로 학습됨 → force_no_bias=True
    # textattack 모델: bias 있음 → 그대로 유지
    no_bias = (task_name == 'sst2' and local_sst2_ckpt is not None
               and os.path.exists(local_sst2_ckpt))
    apply_quantlinear_with_stats(model, force_no_bias=no_bias)
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


def run_all_glue(local_sst2_ckpt=None):
    """전체 GLUE task 실행. Returns dict: task → {accuracy, num_samples}."""
    print(f"Device: {DEVICE}")
    results = {}
    for task in GLUE_TASKS:
        print(f"  [{task.upper():5s}] evaluating...", end=' ', flush=True)
        try:
            acc, n = evaluate_task(task, local_sst2_ckpt=local_sst2_ckpt)
            results[task] = {'accuracy': acc, 'num_samples': n}
            print(f"acc={acc*100:.2f}%  (n={n})")
        except Exception as e:
            print(f"FAILED: {e}")
            results[task] = {'accuracy': None, 'num_samples': 0}
    return results
