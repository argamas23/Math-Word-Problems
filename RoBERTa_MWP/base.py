# baseline_roberta_answer_classification.py

import json, re, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AutoTokenizer, AutoModel, AdamW, get_linear_schedule_with_warmup

# ── 1. Hyperparameters ────────────────────────────────────────
PRETRAINED_MODEL = "hfl/chinese-roberta-wwm-ext"
DATA_PATH        = "data/Math_23K.json"
BATCH_SIZE       = 64
LR               = 1e-5
WEIGHT_DECAY     = 0.01
EPOCHS           = 10
MAX_LEN          = 128
DEVICE           = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── 2. Answer parser ──────────────────────────────────────────
def parse_answer(ans_str: str) -> float:
    s = ans_str.strip()
    if s.endswith('%'):
        return float(s.rstrip('%')) / 100
    try:
        return float(s)
    except:
        expr = re.sub(r'(?<=\d)\s*\(', '*(', s)
        expr = re.sub(r'\)\s*(?=\d)', ')*', expr)
        expr = re.sub(r'\)\s*\(', ')*(', expr)
        expr = re.sub(r'[^0-9\.\+\-\*\/\(\) ]+', '', expr)
        return float(eval(expr))

# ── 3. Dataset for answer classification ───────────────────────
class MathClassDataset(Dataset):
    def __init__(self, path, tokenizer):
        self.tokenizer = tokenizer
        self.examples = []
        buf = []
        with open(path, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                buf.append(line)
                if line == '}':
                    obj = json.loads("".join(buf))
                    text = obj['original_text']
                    ans  = parse_answer(obj['ans'])
                    self.examples.append((text, ans))
                    buf = []

        # build answer class mapping
        unique_ans = sorted({ans for _, ans in self.examples})
        self.ans2id = {a:i for i,a in enumerate(unique_ans)}
        self.id2ans = {i:a for a,i in self.ans2id.items()}

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        text, ans = self.examples[idx]
        enc = self.tokenizer(
            text,
            max_length=MAX_LEN,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            "input_ids":      enc.input_ids.squeeze(0),
            "attention_mask": enc.attention_mask.squeeze(0),
            "label":          torch.tensor(self.ans2id[ans], dtype=torch.long)
        }

# ── 4. Model (Roberta + classifier) ───────────────────────────
class RobertaForAnswerClassification(nn.Module):
    def __init__(self, checkpoint, num_classes):
        super().__init__()
        self.roberta = AutoModel.from_pretrained(checkpoint)
        hidden_size  = self.roberta.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        out = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        cls_vec = out.last_hidden_state[:,0,:]
        h = self.dropout(cls_vec)
        return self.classifier(h)  # [batch, num_classes]

# ── 5. Training / Evaluation ──────────────────────────────────
def train_epoch(model, loader, optimizer, scheduler, loss_fn):
    model.train()
    total_loss = 0.0
    for b in loader:
        optimizer.zero_grad()
        logits = model(b["input_ids"].to(DEVICE), b["attention_mask"].to(DEVICE))
        loss = loss_fn(logits, b["label"].to(DEVICE))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def eval_epoch(model, loader, loss_fn):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for b in loader:
            logits = model(b["input_ids"].to(DEVICE), b["attention_mask"].to(DEVICE))
            total_loss += loss_fn(logits, b["label"].to(DEVICE)).item()
    return total_loss / len(loader)

# ── 6. Main ───────────────────────────────────────────────────
def main():
    print("Device:", DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL)
    ds        = MathClassDataset(DATA_PATH, tokenizer)

    # 80/10/10 split
    total_len = len(ds)
    train_n   = int(0.8 * total_len)
    val_n     = int(0.1 * total_len)
    test_n    = total_len - train_n - val_n
    train_ds, val_ds, test_ds = random_split(ds, [train_n, val_n, test_n])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE)

    model   = RobertaForAnswerClassification(PRETRAINED_MODEL, num_classes=len(ds.ans2id)).to(DEVICE)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    total_steps = len(train_loader) * EPOCHS
    scheduler   = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )

    best_val = float('inf')
    for epoch in range(EPOCHS):
        tr_loss = train_epoch(model, train_loader, optimizer, scheduler, loss_fn)
        va_loss = eval_epoch(model,   val_loader,   loss_fn)
        print(f"Epoch {epoch+1}/{EPOCHS} — Train Loss: {tr_loss:.4f} — Val Loss: {va_loss:.4f}")
        if va_loss < best_val:
            best_val = va_loss
            torch.save(model.state_dict(), "best_answer_cls.pt")
    print("Training complete. Best val loss:", best_val)

    # ── load best & do test accuracy + dump outputs ──────────────
    model.load_state_dict(torch.load("best_answer_cls.pt", map_location=DEVICE))
    model.eval()

    correct = 0
    total   = 0
    with open("testing_output.txt", "w", encoding="utf-8") as fout:
        fout.write("input_text\tpredicted_answer\ttrue_answer\n")
        for idx in test_ds.indices:
            text, true_ans = ds.examples[idx]
            enc = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=MAX_LEN
            ).to(DEVICE)

            with torch.no_grad():
                logits = model(enc.input_ids, enc.attention_mask)
                pred_id = logits.argmax(-1).item()
            pred_ans = ds.id2ans[pred_id]

            fout.write(f"{text}\t{pred_ans}\t{true_ans}\n")

            if abs(pred_ans - true_ans) < 1e-6:
                correct += 1
            total += 1

    acc = correct / total if total > 0 else 0.0
    print(f"Test Accuracy: {acc:.4f}  ({correct}/{total})")
    print("Per‐example results written to testing_output.txt")


if __name__ == "__main__":
    main()
