"""Control split of the image corpus BY DOCUMENT (year/round/category), used to measure leakage.

Purpose: the same CNN and the same OCR route are evaluated on two splits with the SAME number of folds (13):
  - image_folds_v2.csv  : group = competition year (problems shared across categories never straddle folds);
  - image_folds_doc.csv : group = source document (the usual split by source document) -> the same
                          problem from an adjacent category can appear in both training and test.
The difference in scores estimates how much shared problems inflate the estimated performance.

Labels (including expert corrections) are taken from image_folds_v2.csv, so the split is the only difference.

Command:  python src/data_image/build_image_folds_doc.py
Output:   reports/image/v2/image_folds_doc.csv, reports/image/v2/image_folds_doc_report.json
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
V2 = REPO_ROOT / "reports" / "image" / "v2"
ANNOT = REPO_ROOT / "reports" / "image" / "final_image_annotations.csv"


def main() -> int:
    year_folds = pd.read_csv(V2 / "image_folds_v2.csv")
    doc = pd.read_csv(ANNOT)[["task_family_id", "document_group_id"]]
    df = year_folds.drop(columns=["group_id_v2", "fold"]).merge(doc, on="task_family_id")
    n_folds = int(year_folds.fold.nunique())  # same number of folds as the year split (13)

    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=42)
    df["fold"] = -1
    for k, (_, te) in enumerate(sgkf.split(df, df.label, df.document_group_id)):
        df.loc[df.index[te], "fold"] = k
    df["group_id_v2"] = df["document_group_id"]  # column name expected by the modelling scripts
    assert (df.fold >= 0).all()

    # how many candidate same-problem pairs (OCR similarity >= 0.80) end up in different folds
    leak = {}
    pairs_path = V2 / "cross_family_pairs.csv"
    if pairs_path.exists():
        pairs = pd.read_csv(pairs_path)
        f_doc = df.set_index("task_family_id").fold
        f_year = year_folds.set_index("task_family_id").fold
        sp_doc = (pairs.family_a.map(f_doc) != pairs.family_b.map(f_doc))
        sp_year = (pairs.family_a.map(f_year) != pairs.family_b.map(f_year))
        leak = {"candidate_same_task_pairs": int(len(pairs)),
                "pairs_split_across_folds_document_split": int(sp_doc.sum()),
                "pairs_split_across_folds_year_split": int(sp_year.sum())}

    cols = ["task_family_id", "label", "label_original", "year", "round", "category", "group_id_v2", "fold"]
    df[[c for c in cols if c in df]].to_csv(V2 / "image_folds_doc.csv", index=False)
    rep = {"n_families": int(len(df)), "n_folds": n_folds, "group": "document_group_id (year/round/category)",
           "n_groups": int(df.document_group_id.nunique()),
           "fold_sizes": df.groupby("fold").size().astype(int).to_dict(), "leakage": leak}
    (V2 / "image_folds_doc_report.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print(json.dumps(rep, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
