"""
train_glue_w4a8.py
W4A8 QAT fine-tuning for GLUE tasks (MRPC, MNLI).
Same quantization scheme as SST-2 training (WeightQuantSTE, ActQuantSTE, bias=None).

Usage:
    python train_glue_w4a8.py --task mrpc
    python train_glue_w4a8.py --task mnli
"""
import argparse
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup
import warnings
warnings.filterwarnings("ignore")

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# W4A8 Quantization (same as SST-2 training)
# ============================================================
class WeightQuantSTE(torch.autograd.Function):
    W_QMAX = 7
    @staticmethod
    def forward(ctx, w):
        qmax = WeightQuantSTE.W_QMAX
        max_abs = w.abs().max().clamp(min=1e-8)
        scale = max_abs / qmax
        w_int = torch.clamp(torch.round(w / scale), -qmax, qmax)
        return w_int * scale
    @staticmethod
    def backward(ctx, grad):
        return grad

class ActQuantSTE(torch.autograd.Function):
    A_QMAX = 127
    @staticmethod
    def forward(ctx, x):
        qmax = ActQuantSTE.A_QMAX
        max_abs = x.abs().max().clamp(min=1e-8)
        scale = max_abs / qmax
        x_int = torch.clamp(torch.round(x / scale), -qmax, qmax)
        return x_int * scale
    @staticmethod
    def backward(ctx, grad):
        return grad

class QuantLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias=bias)
    def forward(self, x):
        w_q = WeightQuantSTE.apply(self.weight)
        x_q = ActQuantSTE.apply(x)
        return F.linear(x_q, w_q, self.bias)

def apply_qat(model, keep_bias=True):
    """Replace Q/K/V projections with QuantLinear."""
    for layer in model.bert.encoder.layer:
        attn = layer.attention.self
        for name in ["query", "key", "value"]:
            old = getattr(attn, name)
            has_bias = keep_bias and (old.bias is not None)
            ql = QuantLinear(old.in_features, old.out_features, bias=has_bias)
            ql.weight.data.copy_(old.weight.data)
            if has_bias:
                ql.bias.data.copy_(old.bias.data)
            setattr(attn, name, ql)

# ============================================================
# Task configs
# ============================================================
TASK_CONFIGS = {
    'mrpc': {
        'dataset': ('glue', 'mrpc'),
        'train_split': 'train',
        'val_split': 'validation',
        'keys': ('sentence1', 'sentence2'),
        'num_labels': 2,
        'epochs': 10,
        'batch_size': 32,
        'lr': 1e-5,
        'keep_bias': True,      # pretrained bias 유지
        'distill': True,
        'teacher_model': 'textattack/bert-base-uncased-MRPC',
        'distill_alpha': 0.8,   # CE 80%, KL 20%
        'distill_temp':  4.0,
        'eval_steps': 50,       # 50 step마다 중간 평가 (MRPC는 데이터 적어서 자주)
    },
    'mnli': {
        'dataset': ('glue', 'mnli'),
        'train_split': 'train',
        'val_split': 'validation_matched',
        'keys': ('premise', 'hypothesis'),
        'num_labels': 3,
        'epochs': 3,
        'batch_size': 32,
        'lr': 2e-5,
        'keep_bias': True,
        'distill': True,
        'teacher_model': 'textattack/bert-base-uncased-MNLI',
        'distill_alpha': 0.8,
        'distill_temp':  4.0,
        'eval_steps': 500,      # 500 step마다 중간 평가
    },
}

MAX_LEN = 128

def make_loader(data, tokenizer, keys, batch_size, shuffle):
    k1, k2 = keys
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
    return DataLoader(enc, batch_size=batch_size, shuffle=shuffle, collate_fn=collate)

# ============================================================
# Train
# ============================================================
def train(task, seed=42):
    set_seed(seed)
    cfg = TASK_CONFIGS[task]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_distill = cfg.get('distill', False)
    print(f"\nTask: {task.upper()}  |  Device: {device}  |  Distill: {use_distill}")

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    dataset = load_dataset(*cfg['dataset'])
    train_loader = make_loader(dataset[cfg['train_split']], tokenizer,
                               cfg['keys'], cfg['batch_size'], shuffle=True)
    val_loader   = make_loader(dataset[cfg['val_split']],   tokenizer,
                               cfg['keys'], cfg['batch_size'], shuffle=False)

    model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased', num_labels=cfg['num_labels'])
    apply_qat(model, keep_bias=cfg.get('keep_bias', False))
    model.to(device)

    # Teacher 로드 (distillation 사용 시)
    teacher = None
    if use_distill:
        from transformers import AutoModelForSequenceClassification
        teacher = AutoModelForSequenceClassification.from_pretrained(cfg['teacher_model'])
        teacher.to(device)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
        print(f"  Teacher: {cfg['teacher_model']}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=0.01)
    total_steps = len(train_loader) * cfg['epochs']
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps)

    best_acc, best_path = 0.0, f'W4A8_{task.upper()}_best.pt'
    alpha      = cfg.get('distill_alpha', 0.5)
    T          = cfg.get('distill_temp',  4.0)
    eval_steps = cfg.get('eval_steps', None)
    global_step = 0

    def run_eval():
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                preds = model(**batch).logits.argmax(dim=-1)
                correct += (preds == batch['labels']).sum().item()
                total   += batch['labels'].size(0)
        model.train()
        return correct / total

    for epoch in range(cfg['epochs']):
        model.train()
        total_loss = total_ce = total_kl = 0

        for step, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)

            if use_distill:
                with torch.no_grad():
                    t_logits = teacher(
                        input_ids=batch['input_ids'],
                        attention_mask=batch['attention_mask']
                    ).logits
                ce_loss = out.loss
                kl_loss = F.kl_div(
                    F.log_softmax(out.logits / T, dim=-1),
                    F.softmax(t_logits / T, dim=-1),
                    reduction='batchmean'
                ) * (T ** 2)
                loss = alpha * ce_loss + (1 - alpha) * kl_loss
                total_ce += ce_loss.item()
                total_kl += kl_loss.item()
            else:
                loss = out.loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
            global_step += 1

            # 중간 평가
            if eval_steps and global_step % eval_steps == 0:
                acc = run_eval()
                print(f"    [step {global_step}] val_acc={acc*100:.2f}%", end="")
                if acc > best_acc:
                    best_acc = acc
                    torch.save(model.state_dict(), best_path)
                    print(f"  → saved (best={best_acc*100:.2f}%)", end="")
                print()

        # Epoch 끝 평가
        acc = run_eval()
        n = len(train_loader)
        if use_distill:
            print(f"  Epoch {epoch+1}/{cfg['epochs']}  "
                  f"loss={total_loss/n:.4f}  ce={total_ce/n:.4f}  kl={total_kl/n:.4f}  "
                  f"val_acc={acc*100:.2f}%")
        else:
            print(f"  Epoch {epoch+1}/{cfg['epochs']}  "
                  f"loss={total_loss/n:.4f}  val_acc={acc*100:.2f}%")

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), best_path)
            print(f"    → saved (best={best_acc*100:.2f}%)")

    print(f"\nDone. Best acc: {best_acc*100:.2f}%  saved to: {best_path}")
    return best_path, best_acc

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', choices=['mrpc', 'mnli', 'both'], default='both')
    args = parser.parse_args()

    tasks = ['mrpc', 'mnli'] if args.task == 'both' else [args.task]
    for t in tasks:
        train(t)
