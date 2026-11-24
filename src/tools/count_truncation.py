"""Count how many text problems exceed max_len tokens (i.e. are truncated for BERTic / XLM-R).

No GPU needed; only the tokenizers are downloaded (a few MB).
    python src/tools/count_truncation.py
Output: console + reports/ml/text_v2/truncation.json
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATA = REPO_ROOT / "data" / "interim" / "text_expanded" / "problems_ml_ready_v2.jsonl"
OUT = REPO_ROOT / "reports" / "ml" / "text_v2" / "truncation.json"
MAX_LEN = 256

df = pd.read_json(DATA, lines=True)
res = {}
for name in ("classla/bcms-bertic", "xlm-roberta-base"):
    tok = AutoTokenizer.from_pretrained(name)
    n = np.array([len(tok(t, truncation=False)["input_ids"]) for t in df.model_text.astype(str)])
    over = n > MAX_LEN
    res[name] = {"n": int(len(n)), "max_len": MAX_LEN, "n_truncated": int(over.sum()),
                 "pct_truncated": round(100 * over.mean(), 2), "median_tokens": int(np.median(n)),
                 "p95_tokens": int(np.percentile(n, 95)), "max_tokens": int(n.max()),
                 "pct_tokens_lost_in_truncated": round(100 * float((n[over] - MAX_LEN).sum() / n[over].sum()), 1) if over.any() else 0.0}
    print(name, res[name])
OUT.write_text(json.dumps(res, indent=2), encoding="utf-8")
print(f"Output: {OUT}")
