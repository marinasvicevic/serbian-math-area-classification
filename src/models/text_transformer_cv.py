"""Fine-tuning of BERTic / XLM-R on exactly the SAME folds as the TF-IDF models.

Uses reports/ml/text_v2/folds.csv written by text_paper_protocol.py, so the results are directly
comparable with the linear models (same test folds, same corrected t-test). The recipe is FIXED
in advance (no selection on test folds): lr 2e-5, 5 epochs, batch 16, max_len 256, linear warmup
over the first 10% of steps followed by linear decay, weight decay 0.01, gradient clipping 1.0,
AMP fp16 on GPU; the model after the last epoch is used (no early stopping).
A plain PyTorch training loop is used (independent of changes in the HF Trainer API).

Installation: pip install transformers  (+ torch with CUDA)
Models (Hugging Face): classla/bcms-bertic, xlm-roberta-base

Commands (a quick smoke test first is recommended):
    python src/models/text_transformer_cv.py --model classla/bcms-bertic --max-folds 1 --epochs 1   # smoke
    python src/models/text_transformer_cv.py --model classla/bcms-bertic
    python src/models/text_transformer_cv.py --model xlm-roberta-base
Output: reports/ml/text_v2/transformer_<name>/fold_scores.csv, predictions.csv
After an interruption, rerun the same command: finished folds are skipped.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import (AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding,
                          get_linear_schedule_with_warmup, set_seed)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATA = REPO_ROOT / "data" / "interim" / "text_expanded" / "problems_ml_ready_v2.jsonl"
FOLDS = REPO_ROOT / "reports" / "ml" / "text_v2" / "folds.csv"
CLASSES = ["Algebra", "Geometrija", "Teorija brojeva", "Kombinatorika"]


class TextDS(Dataset):
    def __init__(self, enc, labels):
        self.enc, self.labels = enc, labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        item = {k: v[i] for k, v in self.enc.items()}  # lists; the collator pads and builds tensors
        item["labels"] = int(self.labels[i])
        return item


def train_and_predict(model, ds_tr, ds_te, collate, args, seed, device):
    """Plain training loop (without the HF Trainer, whose API changes between versions).
    Recipe: AdamW (weight decay 0.01, none for bias/LayerNorm), linear warmup 10% then linear decay,
    gradient clipping 1.0, AMP fp16 on GPU, fixed number of epochs, no selection on test folds."""
    use_amp = device.type == "cuda"
    dl_tr = DataLoader(ds_tr, batch_size=args.batch, shuffle=True, collate_fn=collate,
                       generator=torch.Generator().manual_seed(seed))
    dl_te = DataLoader(ds_te, batch_size=64, shuffle=False, collate_fn=collate)
    no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight")
    groups = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": 0.01},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    opt = torch.optim.AdamW(groups, lr=args.lr)
    total = len(dl_tr) * args.epochs
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * total), total)
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    model.to(device)
    for ep in range(args.epochs):
        model.train()
        run = 0.0
        for step, batch in enumerate(dl_tr, 1):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                loss = model(**batch).loss
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            opt.zero_grad(set_to_none=True)
            sched.step()
            run += float(loss.item())
        print(f"   epoch {ep + 1}/{args.epochs}: loss={run / max(step, 1):.4f}", flush=True)
    model.eval()
    preds = []
    with torch.no_grad():
        for batch in dl_te:
            batch = {k: v.to(device) for k, v in batch.items() if k != "labels"}
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                logits = model(**batch).logits
            preds.append(logits.float().argmax(-1).cpu().numpy())
    return np.concatenate(preds)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--max-folds", type=int, default=None)
    ap.add_argument("--out-root", type=Path, default=REPO_ROOT / "reports" / "ml" / "text_v2",
                    help="where results are stored (e.g. a Google Drive folder)")
    args = ap.parse_args()

    df = pd.read_json(DATA, lines=True)
    folds = pd.read_csv(FOLDS)
    lab = {c: i for i, c in enumerate(CLASSES)}
    df["y"] = df.oblast.map(lab)
    tok = AutoTokenizer.from_pretrained(args.model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    out = args.out_root / f"transformer_{args.model.replace('/', '_')}"
    out.mkdir(parents=True, exist_ok=True)
    rows, preds = [], []
    if (out / "fold_scores.csv").exists():  # resume after an interruption
        rows = pd.read_csv(out / "fold_scores.csv").to_dict("records")
        preds = pd.read_csv(out / "predictions.csv").to_dict("records")
    done = {(int(x["repeat"]), int(x["fold"])) for x in rows}
    if done:
        print(f"Resuming: {len(done)} folds already done.")
    pairs = folds[["repeat", "fold"]].drop_duplicates().sort_values(["repeat", "fold"]).to_numpy()
    for n, (r, k) in enumerate(pairs):
        if args.max_folds and n >= args.max_folds:
            break
        if (int(r), int(k)) in done:
            continue
        test_ids = set(folds[(folds.repeat == r) & (folds.fold == k)].id)
        te, tr = df[df.id.isin(test_ids)], df[~df.id.isin(test_ids)]
        set_seed(42 + int(r))
        enc = lambda s: tok(list(s), truncation=True, max_length=args.max_len)  # noqa: E731
        ds_tr, ds_te = TextDS(enc(tr.model_text), tr.y.tolist()), TextDS(enc(te.model_text), te.y.tolist())
        model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=len(CLASSES))
        pred = train_and_predict(model, ds_tr, ds_te, DataCollatorWithPadding(tok), args, 42 + int(r), device)
        f1 = f1_score(te.y, pred, average="macro")
        rows.append({"repeat": int(r), "fold": int(k), "n_train": len(tr), "n_test": len(te),
                     "macro_f1": f1, "accuracy": accuracy_score(te.y, pred)})
        preds += [{"repeat": int(r), "fold": int(k), "id": i, "true": CLASSES[t], "pred": CLASSES[p]}
                  for i, t, p in zip(te.id, te.y, pred)]
        print(rows[-1])
        # write after EVERY fold (nothing is lost on interruption)
        pd.DataFrame(preds).to_csv(out / "predictions.csv", index=False)
        pd.DataFrame(rows).to_csv(out / "fold_scores.csv", index=False)
        del model
        torch.cuda.empty_cache()
    s = pd.DataFrame(rows)
    print(f"macro F1 = {s.macro_f1.mean():.4f} ± {s.macro_f1.std(ddof=1):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
