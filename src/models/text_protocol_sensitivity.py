"""Sensitivity of the estimate to the evaluation protocol (Table 4 in the paper).

For ONE fixed configuration (word TF-IDF + LinearSVC, C=0.25, balanced class weights)
macro-F1 is computed under:
  a) a single random 70/15/15 split grouped by exact duplicates, for 10 seeds
     (trained on train+val, tested on test) - shows the spread of a single split;
  b) 5x5 grouped CV with EXACT-duplicate groups (text_hash);
  c) 5x5 grouped CV with NEAR-duplicate groups (group_id_v2).
Because the configuration is fixed there is no inner model selection, so this measures
only the effect of the protocol, not of the model.

Command:
    python src/models/text_protocol_sensitivity.py
Output: reports/ml/text_v2/protocol_sensitivity.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
IN_PATH = REPO_ROOT / "data" / "interim" / "text_expanded" / "problems_ml_ready_v2.jsonl"
OUT = REPO_ROOT / "reports" / "ml" / "text_v2" / "protocol_sensitivity.json"


def model():
    return make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.95, sublinear_tf=True),
        LinearSVC(C=0.25, class_weight="balanced", max_iter=10000, random_state=42),
    )


def f1(y, p):
    return f1_score(y, p, average="macro")


def main() -> int:
    df = pd.read_json(IN_PATH, lines=True)
    X, y = df["model_text"].to_numpy(), df["oblast"].to_numpy()
    res = {}

    # a) single split (exact-duplicate groups, stratified at group level)
    single = []
    G = df.groupby("text_hash", as_index=False).agg(label=("oblast", "first"))
    for seed in range(10):
        tr, rem = train_test_split(G, test_size=0.30, random_state=seed, stratify=G.label)
        _, te = train_test_split(rem, test_size=0.50, random_state=seed, stratify=rem.label)
        test_mask = df.text_hash.isin(te.text_hash).to_numpy()
        m = model().fit(X[~test_mask], y[~test_mask])  # train+val -> test
        single.append(f1(y[test_mask], m.predict(X[test_mask])))
    res["single_split_10_seeds"] = {"mean": round(float(np.mean(single)), 4), "min": round(min(single), 4),
                                    "max": round(max(single), 4), "values": [round(v, 4) for v in single]}

    for name, groups in [("cv5x5_exact_groups", df.text_hash.to_numpy()), ("cv5x5_neardup_groups", df.group_id_v2.to_numpy())]:
        sc = []
        for r in range(5):
            for a, b in StratifiedGroupKFold(5, shuffle=True, random_state=42 + r).split(X, y, groups):
                sc.append(f1(y[b], model().fit(X[a], y[a]).predict(X[b])))
        res[name] = {"mean": round(float(np.mean(sc)), 4), "sd": round(float(np.std(sc, ddof=1)), 4)}

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
