"""Text corpus: model input text and near-duplicate groups (group_id_v2).

What it does
------------
1. Reads the cleaned corpus (`--input`; the final run used
   `data/interim/text_expanded/problems_clean.jsonl`, 2974 records). The input file is NOT modified.
2. Builds the field `model_text`: `clean_text` without the trailing author/source attribution
   (e.g. `\\hfill \\textit{<author>}`, `\\hfill (M361)`, `\\hfill \\textit{Tangenta 75}`).
   The proposer's name or source could act as a shortcut unrelated to the mathematical content.
3. Computes pairwise similarity (TF-IDF over character 4-grams, cosine).
   Pairs with similarity >= --threshold (default 0.70) are linked; connected components
   (together with exact duplicates) define `group_id_v2`. All splits and CV use `group_id_v2`.
4. Reports for MANUAL review:
   - neardup_pairs.csv: all pairs above the threshold (sorted), with labels;
   - neardup_label_conflicts.csv: near-duplicates with DIFFERENT labels (sent to the experts).

Command (from the repository root), as used for the paper:
    python src/data/build_text_v2_groups.py --input data/interim/text_expanded/problems_clean.jsonl \\
        --reviewed reports/data/text_v2/neardup_review.xlsx \\
        --corrections reports/data/text_v2/ispravke_oznaka.xlsx
  (neardup_review.xlsx: manual same-problem decisions, column isti_zadatak = DA/NE (yes/no);
   ispravke_oznaka.xlsx: expert label decisions, column konacna_oznaka = final label)

Output:
    data/interim/text_expanded/problems_ml_ready_v2.jsonl
    reports/data/text_v2/neardup_pairs.csv
    reports/data/text_v2/neardup_label_conflicts.csv
    reports/data/text_v2/groups_v2_report.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.feature_extraction.text import TfidfVectorizer

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
IN_PATH = REPO_ROOT / "data" / "interim" / "text_expanded" / "problems_ml_ready.jsonl"
OUT_PATH = REPO_ROOT / "data" / "interim" / "text_expanded" / "problems_ml_ready_v2.jsonl"
REPORT_DIR = REPO_ROOT / "reports" / "data" / "text_v2"

# Author/source tail: everything from \hfill to the end of the text, and a standalone (M123).
ATTRIBUTION_TAIL = re.compile(r"\\hfill.*$", re.DOTALL)
SOURCE_TAG = re.compile(r"\((?:M|m)\s?\d+\)\s*$")
LEADING_SOURCE_TAG = re.compile(r"^\s*\((?:M|m)\s?\d+\)\s*")  # e.g. "(M567) Neka su ..."
CLASSES = ["Algebra", "Geometrija", "Teorija brojeva", "Kombinatorika"]


def make_model_text(text: str) -> str:
    t = ATTRIBUTION_TAIL.sub("", text)
    t = SOURCE_TAG.sub("", t.strip())
    t = LEADING_SOURCE_TAG.sub("", t)
    return t.strip()


def comparison_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def load(path: Path) -> pd.DataFrame:
    with path.open(encoding="utf-8") as f:
        return pd.DataFrame([json.loads(line) for line in f if line.strip()])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=IN_PATH)
    ap.add_argument("--output", type=Path, default=OUT_PATH)
    ap.add_argument("--report-dir", type=Path, default=REPORT_DIR)
    ap.add_argument("--threshold", type=float, default=0.70)
    ap.add_argument("--reviewed", type=Path, default=None,
                    help="neardup_review.xlsx after manual review (column isti_zadatak = DA/NE, i.e. yes/no)")
    ap.add_argument("--corrections", type=Path, default=None,
                    help="ispravke_oznaka.xlsx (sheet Ispravke: id, trenutna_oznaka = current label, konacna_oznaka = final label)")
    args = ap.parse_args()

    df = load(args.input).reset_index(drop=True)
    df["oblast_original"] = df["oblast"]
    conflicts_json = REPO_ROOT / "reports" / "data" / "text_expanded_label_conflicts.json"
    excluded_ids: set[str] = set()
    if conflicts_json.exists():
        _c = json.loads(conflicts_json.read_text(encoding="utf-8"))
        excluded_ids = {i for g in _c["groups"] for i in g["member_ids"]}

    # Expert label corrections: the source data are NOT modified; corrections apply only to the v2 layer.
    corr_info = {"corrections_file": None}
    if args.corrections:
        cor = pd.read_excel(args.corrections, sheet_name="Ispravke", dtype=str)
        cor["konacna_oznaka"] = cor["konacna_oznaka"].fillna("").str.strip()
        cor = cor[cor["konacna_oznaka"] != ""][["id", "konacna_oznaka"]]
        # Sheet with the 11 groups of identical texts that originally carried different labels.
        sheets = pd.ExcelFile(args.corrections).sheet_names
        if "Identicni_konflikti" in sheets:
            ik = pd.read_excel(args.corrections, sheet_name="Identicni_konflikti", dtype=str)
            ik["konacna_oznaka"] = ik["konacna_oznaka"].fillna("").str.strip()
            ik = ik[ik["konacna_oznaka"] != ""]
            rows_ik = [(i.strip(), lab) for ids, lab in zip(ik["id_jevi"], ik["konacna_oznaka"]) for i in ids.split(",")]
            cor = pd.concat([cor, pd.DataFrame(rows_ik, columns=["id", "konacna_oznaka"])], ignore_index=True)
        bad = sorted(set(cor["konacna_oznaka"]) - set(CLASSES))
        if bad:
            raise SystemExit(f"Invalid labels in the corrections: {bad}. Allowed: {CLASSES}")
        unknown = sorted(set(cor["id"]) - set(df["id"]))
        if unknown and len(df) < 2974:
            print(f"NOTE: {len(unknown)} ids from the corrections are not in the input file (excluded conflicts). "
                  "To include them, run with --input data/interim/text_expanded/problems_clean.jsonl")
            cor = cor[cor["id"].isin(df["id"])]
        elif unknown:
            raise SystemExit(f"Ids from the corrections do not exist in the corpus: {unknown}")
        dup = cor.groupby("id")["konacna_oznaka"].nunique()
        if (dup > 1).any():
            raise SystemExit(f"The same id has different final labels in the corrections: {list(dup[dup > 1].index)}")
        new = dict(zip(cor["id"], cor["konacna_oznaka"]))
        df["oblast"] = [new.get(i, o) for i, o in zip(df["id"], df["oblast"])]
        changed = df[df["oblast"] != df["oblast_original"]]
    # Identical texts must share one label (otherwise a grouped split is meaningless).
    still = df.groupby("text_hash")["oblast"].nunique()
    still = still[still > 1]
    if len(still):
        ids_left = df[df["text_hash"].isin(still.index)]["id"].tolist()
        raise SystemExit(f"{len(still)} groups of identical texts still have different labels: {ids_left}. "
                         "Resolve them in the Identicni_konflikti sheet of the corrections file (or use problems_ml_ready.jsonl).")
    if args.corrections:
        corr_info = {"corrections_file": str(args.corrections), "n_rows_with_decision": int(len(cor)),
                     "n_previously_excluded_conflict_records_returned": int(df["id"].isin(excluded_ids).sum()),
                     "n_labels_changed": int(len(changed)),
                     "changes": [{"id": i, "from": a, "to": b} for i, a, b in
                                 zip(changed["id"], changed["oblast_original"], changed["oblast"])]}
    df["model_text"] = df["clean_text"].map(make_model_text)
    n_tail = int((df["model_text"] != df["clean_text"].str.strip()).sum())

    comp = df["model_text"].map(comparison_text)
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(4, 4), sublinear_tf=True)
    X = vec.fit_transform(comp)
    S = (X @ X.T).tocoo()
    mask = (S.row < S.col) & (S.data >= args.threshold)
    rows, cols, sims = S.row[mask], S.col[mask], S.data[mask]

    # Manual review: pairs marked NE (no) are NOT linked (exact duplicates are always linked below).
    review_info = {"reviewed_file": None}
    if args.reviewed:
        rev = pd.read_excel(args.reviewed, sheet_name="Pregled")
        dec = rev["isti_zadatak"].fillna("").astype(str).str.strip().str.upper()
        n_empty = int((dec == "").sum())
        if n_empty:
            print(f"WARNING: {n_empty} pairs have no decision - treated as DA (same problem).")
        rejected = {frozenset((a, b)) for a, b, d in zip(rev["id_a"], rev["id_b"], dec) if d == "NE"}
        ids = df["id"].to_numpy()
        keep = np.array([frozenset((ids[a], ids[b])) not in rejected for a, b in zip(rows, cols)], dtype=bool)
        review_info = {"reviewed_file": str(args.reviewed), "n_reviewed_pairs": int(len(rev)),
                       "n_marked_DA": int((dec == "DA").sum()), "n_marked_NE": int((dec == "NE").sum()),
                       "n_without_decision_treated_as_DA": n_empty}
        rows, cols, sims = rows[keep], cols[keep], sims[keep]

    # Edges: near-duplicates + exact duplicates (same text_hash).
    exact = df.groupby("text_hash").indices
    er, ec = [], []
    for idx in exact.values():
        idx = list(idx)
        er += idx[:-1]
        ec += idx[1:]
    n = len(df)
    A = coo_matrix(
        (np.ones(len(rows) + len(er)), (np.r_[rows, er].astype(int), np.r_[cols, ec].astype(int))), shape=(n, n)
    )
    n_comp, labels = connected_components(A, directed=False)
    df["group_id_v2"] = [f"g2_{c:05d}" for c in labels]

    pairs = pd.DataFrame(
        {
            "similarity": np.round(sims, 4),
            "id_a": df["id"].values[rows],
            "id_b": df["id"].values[cols],
            "oblast_a": df["oblast"].values[rows],
            "oblast_b": df["oblast"].values[cols],
            "exact_duplicate": df["text_hash"].values[rows] == df["text_hash"].values[cols],
            "text_a": df["model_text"].values[rows],
            "text_b": df["model_text"].values[cols],
        }
    ).sort_values("similarity", ascending=False)
    conflicts = pairs[pairs["oblast_a"] != pairs["oblast_b"]]

    sizes = pd.Series(labels).value_counts()
    args.report_dir.mkdir(parents=True, exist_ok=True)
    if not args.reviewed:  # after the review the pair list is not overwritten
        pairs.to_csv(args.report_dir / "neardup_pairs.csv", index=False, encoding="utf-8")
    conflicts.to_csv(args.report_dir / "neardup_label_conflicts.csv", index=False, encoding="utf-8")
    report = {
        "input": str(args.input),
        "n_records": n,
        "threshold": args.threshold,
        "similarity": "cosine, TF-IDF char_wb 4-grams, sublinear_tf, over model_text (lowercase, whitespace-normalized)",
        "n_records_with_attribution_tail_removed": n_tail,
        "n_pairs_above_threshold": int(len(pairs)),
        "n_pairs_above_threshold_not_exact": int((~pairs["exact_duplicate"]).sum()),
        "n_pairs_with_label_disagreement": int(len(conflicts)),
        "manual_review": review_info,
        "label_corrections": corr_info,
        "class_distribution": {k: int(v) for k, v in df["oblast"].value_counts().items()},
        "n_groups_v1_exact": int(df["text_hash"].nunique()),
        "n_groups_v2": int(n_comp),
        "largest_group_v2": int(sizes.max()),
        "group_size_histogram": {str(k): int(v) for k, v in sizes.value_counts().sort_index().items()},
    }
    (args.report_dir / "groups_v2_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as f:
        for rec in df.to_dict(orient="records"):
            f.write(json.dumps({k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in rec.items()}, ensure_ascii=False) + "\n")

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("\nREQUIRED: manually review", args.report_dir / "neardup_pairs.csv",
          "(at least pairs 0.70-0.85) and send to the experts", args.report_dir / "neardup_label_conflicts.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
