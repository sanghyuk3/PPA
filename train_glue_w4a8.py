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
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup
import warnings
warnings.filterwarnings("ignore")

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
    def __init__(self, in_features, out_features):
        super().__init__(in_features, out_features, bias=False)
    def forward(self, x):
        w_q = WeightQuantSTE.apply(self.weight)
        x_q = ActQuantSTE.apply(x)
        return F.linear(x_q, w_q, None)

def apply_qat(model):
    """Replace Q/K/V projections with QuantLinear."""
    for layer in model.bert.encoder.layer:
        attn = layer.attention.self
        for name in ["query", "key", "value"]:
            old = getattr(attn, name)
            ql = QuantLinear(old.in_features, old.out_features)
            ql.weight.data.copy_(old.weight.data)
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
        'epochs': 5,
        'batch_size': 32,
        'lr': 2e-5,
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
def train(task):
    cfg = TASK_CONFIGS[task]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nTask: {task.upper()}  |  Device: {device}")

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    dataset = load_dataset(*cfg['dataset'])
    train_loader = make_loader(dataset[cfg['train_split']], tokenizer,
                               cfg['keys'], cfg['batch_size'], shuffle=True)
    val_loader   = make_loader(dataset[cfg['val_split']],   tokenizer,
                               cfg['keys'], cfg['batch_size'], shuffle=False)

    model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased', num_labels=cfg['num_labels'])
    apply_qat(model)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=0.01)
    total_steps = len(train_loader) * cfg['epochs']
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps)

    best_acc, best_path = 0.0, f'W4A8_{task.upper()}_best.pt'

    for epoch in range(cfg['epochs']):
        # Train
        model.train()
        total_loss = 0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        # Validate
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                out = model(**batch)
                preds = out.logits.argmax(dim=-1)
                correct += (preds == batch['labels']).sum().item()
                total += batch['labels'].size(0)
        acc = correct / total

        print(f"  Epoch {epoch+1}/{cfg['epochs']}  "
              f"loss={total_loss/len(train_loader):.4f}  val_acc={acc*100:.2f}%")

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
