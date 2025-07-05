# # baseline_roberta_equation_fixed.py

# import json
# import torch
# import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader, random_split
# from transformers import AutoTokenizer, AutoModel, AdamW, get_linear_schedule_with_warmup

# PRETRAINED_MODEL = "hfl/chinese-roberta-wwm-ext"
# DATA_PATH        = "data/Math_23K.json"
# BATCH_SIZE       = 64
# LR               = 1e-5
# WEIGHT_DECAY     = 0.01
# EPOCHS           = 10
# MAX_LEN          = 128
# DEVICE           = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# class EqDataset(Dataset):
#     def __init__(self, path, tokenizer):
#         self.examples = []
#         with open(path, encoding='utf-8') as f:
#             buf = []
#             for line in f:
#                 if not line.strip():
#                     continue
#                 buf.append(line)
#                 if line.strip() == '}':
#                     obj = json.loads(''.join(buf))
#                     self.examples.append((
#                         obj['original_text'],
#                         obj['equation'].strip()
#                     ))
#                     buf = []

#         # build your class↔string maps
#         eqs = sorted({ eq for _, eq in self.examples })
#         self.eq2id = { e:i for i,e in enumerate(eqs) }
#         self.id2eq = { i:e for e,i in self.eq2id.items() }

#         self.tokenizer = tokenizer

#     def __len__(self):
#         return len(self.examples)

#     def __getitem__(self, i):
#         text, eq = self.examples[i]
#         enc = self.tokenizer(
#             text,
#             max_length=MAX_LEN,
#             padding='max_length',
#             truncation=True,
#             return_tensors='pt'
#         )
#         return {
#             "input_ids":      enc.input_ids.squeeze(0),
#             "attention_mask": enc.attention_mask.squeeze(0),
#             "label":          torch.tensor(self.eq2id[eq], dtype=torch.long)
#         }


# class RobertaEqOnly(nn.Module):
#     def __init__(self, checkpoint, num_eq):
#         super().__init__()
#         self.roberta = AutoModel.from_pretrained(checkpoint)
#         hidden       = self.roberta.config.hidden_size
#         self.dropout = nn.Dropout(0.1)
#         self.head    = nn.Linear(hidden, num_eq)

#     def forward(self, input_ids, attention_mask):
#         out = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
#         h   = self.dropout(out.last_hidden_state[:,0,:])
#         return self.head(h)   # [batch, num_eq]


# def train_epoch(model, loader, optimizer, scheduler, loss_fn):
#     model.train()
#     total = 0.0
#     for b in loader:
#         optimizer.zero_grad()
#         logits = model(
#             b["input_ids"].to(DEVICE),
#             b["attention_mask"].to(DEVICE)
#         )
#         loss = loss_fn(logits, b["label"].to(DEVICE))
#         loss.backward()
#         nn.utils.clip_grad_norm_(model.parameters(), 1.0)
#         optimizer.step()
#         scheduler.step()
#         total += loss.item()
#     return total / len(loader)


# def eval_epoch(model, loader, loss_fn):
#     model.eval()
#     total = 0.0
#     with torch.no_grad():
#         for b in loader:
#             logits = model(
#                 b["input_ids"].to(DEVICE),
#                 b["attention_mask"].to(DEVICE)
#             )
#             total += loss_fn(logits, b["label"].to(DEVICE)).item()
#     return total / len(loader)


# def main():
#     # ── prepare data ────────────────────────────────────────────
#     tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL)
#     ds        = EqDataset(DATA_PATH, tokenizer)

#     # 80/10/10 split
#     total_len = len(ds)
#     train_n   = int(0.8 * total_len)
#     val_n     = int(0.1 * total_len)
#     test_n    = total_len - train_n - val_n
#     train_ds, val_ds, test_ds = random_split(ds, [train_n, val_n, test_n])

#     train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
#     val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE)
#     # we’ll handle test_ds manually below

#     # ── build model ─────────────────────────────────────────────
#     model   = RobertaEqOnly(PRETRAINED_MODEL, num_eq=len(ds.eq2id)).to(DEVICE)
#     loss_fn = nn.CrossEntropyLoss()
#     optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

#     total_steps = len(train_loader) * EPOCHS
#     scheduler   = get_linear_schedule_with_warmup(
#         optimizer,
#         num_warmup_steps=int(0.1 * total_steps),
#         num_training_steps=total_steps
#     )

#     # ── train & validate ────────────────────────────────────────
#     best_val = float('inf')
#     for epoch in range(EPOCHS):
#         tr_loss = train_epoch(model, train_loader, optimizer, scheduler, loss_fn)
#         val_loss = eval_epoch(model, val_loader, loss_fn)
#         print(f"Epoch {epoch+1}/{EPOCHS} — Train Loss: {tr_loss:.4f} — Val Loss: {val_loss:.4f}")
#         if val_loss < best_val:
#             best_val = val_loss
#             torch.save(model.state_dict(), "best_eq_only.pt")

#     print("Training complete. Best val loss:", best_val)

#     # ── load best & do test accuracy + dump outputs ─────────────
#     model.load_state_dict(torch.load("best_eq_only.pt", map_location=DEVICE))
#     model.eval()

#     correct = 0
#     total   = 0

#     with open("testing_output.txt", "w", encoding="utf-8") as fout:
#         fout.write("input_text\tpredicted_equation\ttrue_equation\n")
#         # test_ds is a Subset: .indices maps to original ds.examples
#         for idx in test_ds.indices:
#             text, true_eq = ds.examples[idx]
#             enc = tokenizer(
#                 text,
#                 return_tensors="pt",
#                 truncation=True,
#                 max_length=MAX_LEN
#             ).to(DEVICE)

#             with torch.no_grad():
#                 logits  = model(enc.input_ids, enc.attention_mask)
#                 pred_id = logits.argmax(-1).item()
#             pred_eq = ds.id2eq[pred_id]

#             fout.write(f"{text}\t{pred_eq}\t{true_eq}\n")

#             if pred_eq == true_eq:
#                 correct += 1
#             total += 1

#     acc = correct / total if total > 0 else 0.0
#     print(f"Test Accuracy: {acc:.4f}  ({correct}/{total})")
#     print("Per‐example results written to testing_output.txt")


# if __name__ == "__main__":
#     main()import json
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AutoTokenizer, AutoModel, AdamW, get_linear_schedule_with_warmup

PRETRAINED_MODEL = "hfl/chinese-roberta-wwm-ext"
DATA_PATH        = "data/Math_23K.json"
BATCH_SIZE       = 64
LR               = 1e-5
WEIGHT_DECAY     = 0.2   # increased weight decay
EPOCHS           = 10
MAX_LEN          = 128
DEVICE           = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EqDataset(Dataset):
    def __init__(self, path, tokenizer):
        self.examples = []
        with open(path, encoding='utf-8') as f:
            buf = []
            for line in f:
                if not line.strip():
                    continue
                buf.append(line)
                if line.strip() == '}':
                    obj = json.loads(''.join(buf))
                    text = obj['original_text']
                    eq   = obj['equation'].strip()
                    self.examples.append((text, eq))
                    buf = []

        # build full-equation classes
        eqs = sorted({ eq for _, eq in self.examples })
        self.eq2id = { e:i for i,e in enumerate(eqs) }
        self.id2eq = { i:e for e,i in self.eq2id.items() }

        # build character-level vocab for equations
        chars = sorted({ ch for _, eq in self.examples for ch in eq })
        self.char2id = { ch:i+1 for i,ch in enumerate(chars) }  # reserve 0 for padding
        self.id2char = { i+1:ch for i,ch in enumerate(chars) }
        self.char2id['<pad>'] = 0
        self.id2char[0] = '<pad>'

        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        text, eq = self.examples[i]
        # text encoding
        enc = self.tokenizer(
            text,
            max_length=MAX_LEN,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        # equation as class label
        cls_label = torch.tensor(self.eq2id[eq], dtype=torch.long)
        # equation as char-token sequence (not used in this model yet)
        eq_ids = [ self.char2id.get(ch, 0) for ch in eq ]
        eq_ids = eq_ids[:MAX_LEN]  # truncate
        eq_ids += [0] * (MAX_LEN - len(eq_ids))  # pad to MAX_LEN
        eq_tensor = torch.tensor(eq_ids, dtype=torch.long)

        return {
            "input_ids":      enc.input_ids.squeeze(0),
            "attention_mask": enc.attention_mask.squeeze(0),
            "cls_label":      cls_label,
            "eq_tokens":      eq_tensor,
        }

class RobertaEqOnly(nn.Module):
    def __init__(self, checkpoint, num_eq):
        super().__init__()
        self.roberta = AutoModel.from_pretrained(checkpoint)
        hidden       = self.roberta.config.hidden_size
        self.dropout = nn.Dropout(0.2)  # increased dropout
        self.head    = nn.Linear(hidden, num_eq)

    def forward(self, input_ids, attention_mask):
        out = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        h   = self.dropout(out.last_hidden_state[:,0,:])
        return self.head(h)   # [batch, num_eq]


def train_epoch(model, loader, optimizer, scheduler, loss_fn):
    model.train()
    total_loss = 0.0
    for b in loader:
        optimizer.zero_grad()
        logits = model(
            b["input_ids"].to(DEVICE),
            b["attention_mask"].to(DEVICE)
        )
        loss = loss_fn(logits, b["cls_label"].to(DEVICE))
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
            logits = model(
                b["input_ids"].to(DEVICE),
                b["attention_mask"].to(DEVICE)
            )
            total_loss += loss_fn(logits, b["cls_label"].to(DEVICE)).item()
    return total_loss / len(loader)

def main():
    print("Device:", DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL)
    ds        = EqDataset(DATA_PATH, tokenizer)

    # 80/10/10 splits
    total = len(ds)
    train_n = int(0.8 * total)
    val_n   = int(0.1 * total)
    test_n  = total - train_n - val_n
    train_ds, val_ds, test_ds = random_split(ds, [train_n, val_n, test_n])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE)

    model   = RobertaEqOnly(PRETRAINED_MODEL, num_eq=len(ds.eq2id)).to(DEVICE)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    total_steps = len(train_loader) * EPOCHS
    scheduler   = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )

    # early stopping setup
    best_val = float('inf')
    patience = 2
    epochs_no_improve = 0

    for epoch in range(EPOCHS):
        tr_loss = train_epoch(model, train_loader, optimizer, scheduler, loss_fn)
        va_loss = eval_epoch(model,   val_loader,   loss_fn)
        print(f"Epoch {epoch+1}/{EPOCHS} — Train Loss: {tr_loss:.4f} — Val Loss: {va_loss:.4f}")

        if va_loss < best_val:
            best_val = va_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), "best_eq_only.pt")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"No val improvement for {patience} epochs; stopping early.")
                break

    print("Training complete. Best val loss:", best_val)

    # ── test accuracy + output dump ─────────────────────────────
    model.load_state_dict(torch.load("best_eq_only.pt", map_location=DEVICE))
    model.eval()
    correct, total = 0, 0
    with open("testing_output1.txt", "w", encoding='utf-8') as fout:
        fout.write("input_text\tpredicted_equation\ttrue_equation\n")
        for idx in test_ds.indices:
            text, true_eq = ds.examples[idx]
            enc = tokenizer(text, return_tensors='pt', truncation=True, max_length=MAX_LEN).to(DEVICE)
            with torch.no_grad():
                logits  = model(enc.input_ids, enc.attention_mask)
                pred_id = logits.argmax(-1).item()
            pred_eq = ds.id2eq[pred_id]
            fout.write(f"{text}\t{pred_eq}\t{true_eq}\n")
            if pred_eq == true_eq:
                correct += 1
            total += 1

    acc = correct / total if total > 0 else 0.0
    print(f"Test Accuracy: {acc:.4f} ({correct}/{total})")
    print("Per-example results in testing_output1.txt")

if __name__ == "__main__":
    main()
