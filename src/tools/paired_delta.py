"""Paired group bootstrap of the difference in macro-F1 between two sets of predictions for the SAME items.

Each prediction set is given as  file:prediction_column  (the file must contain the family/problem id).
Bootstrap groups (competition years for Kangaroo) and true labels are taken from the --groups file.

Examples (run from the repository root):
  text, Claude vs. the reference SVM (first CV repetition, near-duplicate groups):
    python src/tools/paired_delta.py --a reports/ml/text_v2/llm_anthropic_claude-opus-5-5_0shot_text-v2/predictions.csv:pred --b reports/ml/text_v2/oof_predictions_rep0.csv:pred_word_svm --groups reports/ml/text_v2/oof_predictions_rep0.csv
  document split vs. the same split with purged training data (year groups, as in Table 6):
    python src/tools/paired_delta.py --a reports/image/v2/ocr_text_cv_docsplit/oof_predictions.csv:pred_ocr_both --b reports/image/v2/ocr_text_cv_docsplit_purged/oof_predictions.csv:pred_ocr_both
Output: console (difference A - B with a 95% percentile CI)
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
GROUPS = REPO_ROOT / "reports" / "image" / "v2" / "image_folds_v2.csv"


def load(spec: str) -> pd.Series:
    path, col = spec.rsplit(":", 1)
    d = pd.read_csv(path, dtype=str)
    idc = "task_family_id" if "task_family_id" in d else "id"
    return d.set_index(idc)[col]


def macro_f1(cm: np.ndarray) -> float:
    """cm is 4x5: column 4 = invalid answer (INVALID). Such an answer is a false negative for the true area
    and a false positive for no area - identical to sklearn f1_score(labels=<the 4 areas>)."""
    tp = np.diag(cm[:, :4]).astype(float)
    p, t = cm[:, :4].sum(0), cm.sum(1)
    f = np.divide(2 * tp, p + t, out=np.zeros_like(tp), where=(p + t) > 0)
    return float(f.mean())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--groups", type=Path, default=GROUPS,
                    help="file with an id (task_family_id or id), the true label (label or oblast) and group_id_v2")
    ap.add_argument("--n", type=int, default=5000)
    args = ap.parse_args()
    gdf = pd.read_csv(args.groups, dtype=str)
    gdf = gdf.set_index("task_family_id" if "task_family_id" in gdf else "id")
    if "label" not in gdf:
        gdf["label"] = gdf["oblast"]
    a, b = load(args.a), load(args.b)
    ids = gdf.index.intersection(a.index).intersection(b.index)
    classes = sorted(gdf.loc[ids, "label"].unique())
    if len(classes) != 4:
        raise SystemExit(f"Expected 4 classes, found: {classes}")
    k = {c: i for i, c in enumerate(classes)}
    y = gdf.loc[ids, "label"].map(k).to_numpy()
    # invalid or missing answer -> index 4 (counted as an error, never dropped)
    pa, pb = a.loc[ids].map(k).fillna(4).astype(int).to_numpy(), b.loc[ids].map(k).fillna(4).astype(int).to_numpy()
    g = gdf.loc[ids, "group_id_v2"].to_numpy()
    ug = np.unique(g)
    # one confusion matrix per group -> each bootstrap sample is just a sum of matrices
    def cms(p):
        out = np.zeros((len(ug), 4, 5), dtype=int)
        for gi, gg in enumerate(ug):
            m = g == gg
            np.add.at(out[gi], (y[m], p[m]), 1)
        return out
    ca, cb = cms(pa), cms(pb)
    d0 = macro_f1(ca.sum(0)) - macro_f1(cb.sum(0))
    rng = np.random.default_rng(42)
    D = []
    for _ in range(args.n):
        s = rng.integers(0, len(ug), len(ug))
        D.append(macro_f1(ca[s].sum(0)) - macro_f1(cb[s].sum(0)))
    lo, hi = np.percentile(D, [2.5, 97.5])
    print(f"n = {len(ids)} items, {len(ug)} groups")
    print(f"macro-F1 A = {macro_f1(ca.sum(0)):.4f}, B = {macro_f1(cb.sum(0)):.4f}")
    print(f"difference A - B = {d0:+.4f}, 95% CI [{lo:+.4f}, {hi:+.4f}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
