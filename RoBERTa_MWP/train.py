import json, re
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AutoTokenizer, AutoModel, AdamW, get_linear_schedule_with_warmup

# ── 1. Hyperparameters ─────────────────────────────────────────
PRETRAINED_MODEL = "hfl/chinese-roberta-wwm-ext"
DATA_PATH        = "data/Math_23K_trunc.json"
BATCH_SIZE       = 64
LR_FULL          = 1e-5
WEIGHT_DECAY     = 0.2   # increased weight decay
EPOCHS           = 10
MAX_LEN          = 128
DEVICE           = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── 2. Answer parser ────────────────────────────────────────────
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

# ── 3. Dataset with vocab for original and equation tokens ──────
class MathEqDataset(Dataset):
    def __init__(self, path, tokenizer):
        self.tokenizer = tokenizer
        self.examples = []  # list of (text, ans, eq_str)
        with open(path, encoding='utf-8') as f:
            buf = []
            for line in f:
                if not line.strip(): continue
                buf.append(line)
                if line.strip() == '}':
                    obj = json.loads(''.join(buf))
                    self.examples.append((
                        obj['original_text'],
                        parse_answer(obj['ans']),
                        obj['equation'].strip()
                    ))
                    buf = []
        # numeric-answer classes
        unique_ans = sorted({ans for _, ans, _ in self.examples})
        self.ans2id = {a:i for i,a in enumerate(unique_ans)}
        self.id2ans = {i:a for a,i in self.ans2id.items()}
        # equation-string classes
        unique_eq  = sorted({eq for _, _, eq in self.examples})
        self.eq2id   = {e:i for i,e in enumerate(unique_eq)}
        self.id2eq   = {i:e for e,i in self.eq2id.items()}
        # char-level vocabulary for eq
        chars = sorted({ch for _,_,eq in self.examples for ch in eq})
        self.char2id = {ch:i+1 for i,ch in enumerate(chars)}  # 0 pad
        self.char2id['<pad>'] = 0
        self.id2char = {i:ch for ch,i in self.char2id.items()}

    def __len__(self): return len(self.examples)

    def __getitem__(self, idx):
        text, ans, eq = self.examples[idx]
        enc = self.tokenizer(
            text,
            max_length=MAX_LEN,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        ans_lbl = torch.tensor(self.ans2id[ans], dtype=torch.long)
        eq_lbl  = torch.tensor(self.eq2id[eq],  dtype=torch.long)
        eq_ids = [self.char2id.get(ch, 0) for ch in eq][:MAX_LEN]
        eq_ids += [0] * (MAX_LEN - len(eq_ids))
        eq_tokens = torch.tensor(eq_ids, dtype=torch.long)
        return {
            'input_ids':      enc.input_ids.squeeze(0),
            'attention_mask': enc.attention_mask.squeeze(0),
            'ans_label':      ans_lbl,
            'eq_label':       eq_lbl,
            'eq_tokens':      eq_tokens,
        }

# ── 4. Dual classification model ───────────────────────────────
class RobertaDualClass(nn.Module):
    def __init__(self, checkpoint, num_ans, num_eq):
        super().__init__()
        self.roberta   = AutoModel.from_pretrained(checkpoint)
        hidden_size    = self.roberta.config.hidden_size
        self.dropout   = nn.Dropout(0.2)
        self.ans_head  = nn.Linear(hidden_size, num_ans)
        self.eq_head   = nn.Linear(hidden_size, num_eq)

    def forward(self, input_ids, attention_mask):
        out     = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        cls_vec = out.last_hidden_state[:,0,:]
        h       = self.dropout(cls_vec)
        return self.ans_head(h), self.eq_head(h)

# ── 5. Train / Eval Loops ─────────────────────────────────────
def train_epoch(model, loader, optimizer, scheduler, ans_loss_fn, eq_loss_fn):
    model.train()
    tot_ans = tot_eq = tot = 0.0
    for b in loader:
        optimizer.zero_grad()
        ans_logits, eq_logits = model(
            b['input_ids'].to(DEVICE),
            b['attention_mask'].to(DEVICE)
        )
        loss_ans = ans_loss_fn(ans_logits, b['ans_label'].to(DEVICE))
        loss_eq  = eq_loss_fn(eq_logits,  b['eq_label'].to(DEVICE))
        loss     = loss_ans + loss_eq
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        tot_ans += loss_ans.item()
        tot_eq  += loss_eq.item()
        tot     += loss.item()
    n = len(loader)
    return tot_ans/n, tot_eq/n, tot/n


def eval_epoch(model, loader, ans_loss_fn, eq_loss_fn):
    model.eval()
    tot_ans = tot_eq = tot = 0.0
    with torch.no_grad():
        for b in loader:
            ans_logits, eq_logits = model(
                b['input_ids'].to(DEVICE),
                b['attention_mask'].to(DEVICE)
            )
            loss_ans = ans_loss_fn(ans_logits, b['ans_label'].to(DEVICE))
            loss_eq  = eq_loss_fn(eq_logits,  b['eq_label'].to(DEVICE))
            tot_ans += loss_ans.item()
            tot_eq  += loss_eq.item()
            tot     += (loss_ans + loss_eq).item()
    n = len(loader)
    return tot_ans/n, tot_eq/n, tot/n

# ── 6. Main ───────────────────────────────────────────────────
def main():
    print("Device:", DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL)
    ds        = MathEqDataset(DATA_PATH, tokenizer)

    total   = len(ds)
    train_n = int(0.8 * total)
    val_n   = int(0.1 * total)
    test_n  = total - train_n - val_n
    train_ds, val_ds, test_ds = random_split(ds, [train_n, val_n, test_n])
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE)

    model = RobertaDualClass(
        PRETRAINED_MODEL,
        num_ans=len(ds.ans2id),
        num_eq =len(ds.eq2id)
    ).to(DEVICE)

    ans_loss_fn = nn.CrossEntropyLoss()
    eq_loss_fn  = nn.CrossEntropyLoss()
    optimizer   = AdamW(model.parameters(), lr=LR_FULL, weight_decay=WEIGHT_DECAY)
    total_steps = len(train_loader) * EPOCHS
    scheduler   = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )

    best_val = float('inf')
    patience = 2
    no_imp   = 0
    for epoch in range(EPOCHS):
        if epoch == 0:
            for p in model.roberta.parameters(): p.requires_grad = False
        if epoch == 1:
            for p in model.roberta.parameters(): p.requires_grad = True

        tr_ans, tr_eq, tr_all = train_epoch(model, train_loader, optimizer, scheduler, ans_loss_fn, eq_loss_fn)
        va_ans, va_eq, va_all = eval_epoch (model,   val_loader,   ans_loss_fn,   eq_loss_fn)
        print(f"Epoch {epoch+1}/{EPOCHS} — ans: {tr_ans:.4f}/{va_ans:.4f}  eq: {tr_eq:.4f}/{va_eq:.4f}  total: {tr_all:.4f}/{va_all:.4f}")

        if va_all < best_val:
            best_val = va_all
            no_imp   = 0
        else:
            no_imp += 1
            if no_imp >= patience:
                print(f"No val improvement for {patience} epochs; stopping.")
                break

    print("Training complete. Best combined CE loss:", best_val)

    # ── 7. Testing ───────────────────────────────────────────────
    model.eval()
    correct_ans = correct_eq = 0
    total = 0
    with open("testing_output.txt", "w", encoding='utf-8') as fout:
        fout.write("text\tpred_ans\ttrue_ans\tpred_eq\ttrue_eq\n")
        for idx in test_ds.indices:
            text, ans, eq = ds.examples[idx]
            enc = tokenizer(text, return_tensors='pt', truncation=True, max_length=MAX_LEN).to(DEVICE)
            with torch.no_grad():
                ans_logits, eq_logits = model(enc.input_ids, enc.attention_mask)
                pa = ans_logits.argmax(-1).item()
                pe = eq_logits.argmax(-1).item()
            pred_ans = ds.id2ans[pa]
            pred_eq  = ds.id2eq[pe]
            fout.write(f"{text}\t{pred_ans}\t{ans}\t{pred_eq}\t{eq}\n")
            if pred_ans == ans:   correct_ans += 1
            if pred_eq  == eq:    correct_eq  += 1
            total += 1
    print(f"Test accuracy ans: {correct_ans/total:.4f}  eq: {correct_eq/total:.4f}")

if __name__ == "__main__":
    main()