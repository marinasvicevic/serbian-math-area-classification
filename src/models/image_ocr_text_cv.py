"""OCR route: OCR text -> word+char TF-IDF + linear SVM on the Kangaroo corpus.

Same Kangaroo problems, same labels and same folds (image_folds_v2.csv) as the CNN - only the
input is OCR text instead of pixels.

Variants (all at the family level):
  ocr_lat   : Latin-script OCR only
  ocr_cyr   : Cyrillic-script OCR only (transliterated to Latin script)
  ocr_both  : sum of the decision scores (decision_function) of both versions -> argmax
              (analogous to averaging the softmax outputs of the CNN)

Training: in every fold the model is trained on BOTH versions of all training families
(2 examples per family, as for the CNN). Hyperparameters: inner group CV on the training
fold (by group_id_v2), small grid for C.

Command:
    python src/models/image_ocr_text_cv.py

Control experiments (leakage):
    --purge-pairs reports/image/v2/cross_family_pairs_review.xlsx
        for every test fold, all families that are the same problem (DA or no decision, as when
        building the groups) as some test family are removed from TRAINING; the test part is unchanged.
    --calendar-years
        folds = actual calendar years (repeated problems are not moved); used together with --purge-pairs.
    Examples:
      document split + purged training data:
        python src/models/image_ocr_text_cv.py --folds reports/image/v2/image_folds_doc.csv --purge-pairs reports/image/v2/cross_family_pairs_review.xlsx --out reports/image/v2/ocr_text_cv_docsplit_purged
      training-size control (copies stay, the same number of random other families is removed):
        python src/models/image_ocr_text_cv.py --folds reports/image/v2/image_folds_doc.csv --purge-pairs reports/image/v2/cross_family_pairs_review.xlsx --random-control --out reports/image/v2/ocr_text_cv_docsplit_randomremoved
      calendar years + purged training data:
        python src/models/image_ocr_text_cv.py --calendar-years --purge-pairs reports/image/v2/cross_family_pairs_review.xlsx --out reports/image/v2/ocr_text_cv_calendar_purged

Output: reports/image/v2/ocr_text_cv/  (fold_scores.csv, oof_predictions.csv, summary.json)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
FOLDS = REPO_ROOT / "reports" / "image" / "v2" / "image_folds_v2.csv"
OCR = REPO_ROOT / "data" / "image_interim" / "ocr" / "kengur_ocr_v1.jsonl"
OUT = REPO_ROOT / "reports" / "image" / "v2" / "ocr_text_cv"
CLASSES = ["Algebra", "Geometrija", "Teorija brojeva", "Logika"]
C_GRID = [0.005, 0.01, 0.02, 0.03, 0.06, 0.125, 0.25]  # extended downwards: in a first run 0.06 (then the grid edge) was selected in all 13 folds


def make_model(C: float) -> Pipeline:
    feats = FeatureUnion([
        ("w", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.95, sublinear_tf=True)),
        ("c", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True)),
    ])
    return Pipeline([("v", feats), ("clf", LinearSVC(C=C, class_weight="balanced", max_iter=10000))])


def stack(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Two versions per family -> 2 training examples."""
    X = np.r_[df.text_lat.to_numpy(), df.text_cyr.to_numpy()]
    y = np.r_[df.label.to_numpy(), df.label.to_numpy()]
    g = np.r_[df.group_id_v2.to_numpy(), df.group_id_v2.to_numpy()]
    return X, y, g


def scores(model: Pipeline, texts: np.ndarray) -> np.ndarray:
    s = model.decision_function(texts)
    order = [list(model.classes_).index(c) for c in CLASSES]
    return s[:, order]


def macro_f1(y, p):
    return f1_score(y, p, labels=CLASSES, average="macro", zero_division=0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=Path, default=FOLDS)
    ap.add_argument("--ocr", type=Path, default=OCR)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--purge-pairs", type=Path, default=None,
                    help="cross_family_pairs_review.xlsx: remove from training the problems that also occur in test")
    ap.add_argument("--calendar-years", action="store_true", help="fold = calendar year of the problem")
    ap.add_argument("--random-control", action="store_true",
                    help="with --purge-pairs: copies STAY in training and the same number of random other "
                         "families is removed instead (measures only the effect of a smaller training set)")
    args = ap.parse_args()

    df = pd.read_csv(args.folds, dtype={"task_family_id": str}).merge(pd.read_json(args.ocr, lines=True), on="task_family_id")
    if args.calendar_years:
        years = sorted(df.year.unique())
        df["fold"] = df.year.map({y: i for i, y in enumerate(years)})
        df["group_id_v2"] = "cal_" + df.year.astype(str)
    comp = pd.Series(-1, index=df.task_family_id)
    if args.purge_pairs:
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components

        parts = []
        for sh in ("Medjugodisnji", "Ista_godina"):
            try:
                d = pd.read_excel(args.purge_pairs, sheet_name=sh, dtype=str)
            except ValueError:
                continue
            d["odluka"] = d["isti_zadatak"].fillna("").astype(str).str.strip().str.upper()
            parts.append(d[d.odluka != "NE"][["family_a", "family_b"]])
        same = pd.concat(parts, ignore_index=True)
        same = same[same.family_a.isin(comp.index) & same.family_b.isin(comp.index)]
        ix = {f: i for i, f in enumerate(df.task_family_id)}
        G = coo_matrix((np.ones(len(same)), ([ix[a] for a in same.family_a], [ix[b] for b in same.family_b])),
                       shape=(len(df), len(df)))
        _, c = connected_components(G, directed=False)
        comp = pd.Series(c, index=df.task_family_id)
        print(f"Same-problem pairs used for purging: {len(same)}")
    df["_comp"] = df.task_family_id.map(comp).to_numpy()
    df["text_lat"] = df.ocr_latin.fillna("").str.lower()
    df["text_cyr"] = df.ocr_cyrillic_translit.fillna("").str.lower()
    preds = {k: np.empty(len(df), dtype=object) for k in ("ocr_lat", "ocr_cyr", "ocr_both")}
    rows = []
    for fold in sorted(df.fold.unique()):
        tr, te = df[df.fold != fold], df[df.fold == fold]
        assert not set(tr.group_id_v2) & set(te.group_id_v2)
        n_purged = 0
        if args.purge_pairs:
            sizes = df._comp.value_counts()
            test_comps = {c for c in te._comp if sizes[c] > 1}
            keep = ~tr._comp.isin(test_comps)
            n_purged = int((~keep).sum())
            if args.random_control:
                pool = tr.index[keep.to_numpy()]
                drop = np.random.default_rng(1000 + int(fold)).choice(pool, size=min(n_purged, len(pool)), replace=False)
                tr = tr.drop(index=drop)
            else:
                tr = tr[keep]
        Xtr, ytr, gtr = stack(tr)
        # inner group CV for C
        if tr.group_id_v2.nunique() < 3:  # only for tiny test sets; the real corpus has 12 training groups
            gtr = np.r_[tr.task_family_id.to_numpy(), tr.task_family_id.to_numpy()]
        inner = GroupKFold(n_splits=min(4, len(set(gtr))))
        best_C, best = C_GRID[0], -1.0
        for C in C_GRID:
            sc = [macro_f1(ytr[b], make_model(C).fit(Xtr[a], ytr[a]).predict(Xtr[b])) for a, b in inner.split(Xtr, ytr, gtr)]
            if np.mean(sc) > best + 1e-12:
                best, best_C = float(np.mean(sc)), C
        model = make_model(best_C).fit(Xtr, ytr)
        s_lat, s_cyr = scores(model, te.text_lat.to_numpy()), scores(model, te.text_cyr.to_numpy())
        idx = te.index.to_numpy()
        preds["ocr_lat"][idx] = np.array(CLASSES)[s_lat.argmax(1)]
        preds["ocr_cyr"][idx] = np.array(CLASSES)[s_cyr.argmax(1)]
        preds["ocr_both"][idx] = np.array(CLASSES)[(s_lat + s_cyr).argmax(1)]
        for k in preds:
            rows.append({"fold": int(fold), "group": te.group_id_v2.iloc[0], "variant": k, "n_test": len(te),
                         "C": best_C, "n_train_purged": n_purged, "macro_f1": macro_f1(te.label, preds[k][idx]),
                         "accuracy": accuracy_score(te.label, preds[k][idx])})
        print(f"fold {fold} ({te.group_id_v2.iloc[0]}): C={best_C} " +
              " ".join(f"{k}={rows[-3 + i]['macro_f1']:.3f}" for i, k in enumerate(preds)))

    args.out.mkdir(parents=True, exist_ok=True)
    fs = pd.DataFrame(rows)
    fs.to_csv(args.out / "fold_scores.csv", index=False)
    out = df[["task_family_id", "label", "year", "category", "group_id_v2", "fold"]].copy()
    for k, v in preds.items():
        out[f"pred_{k}"] = v
    out["scripts_agree"] = out.pred_ocr_lat == out.pred_ocr_cyr
    out.to_csv(args.out / "oof_predictions.csv", index=False)
    summary = {k: {"macro_f1_pooled": round(macro_f1(df.label, v), 4),
                   "macro_f1_fold_mean": round(fs[fs.variant == k].macro_f1.mean(), 4),
                   "macro_f1_fold_sd": round(fs[fs.variant == k].macro_f1.std(ddof=1), 4),
                   "accuracy_pooled": round(accuracy_score(df.label, v), 4),
                   "per_class_f1": dict(zip(CLASSES, np.round(f1_score(df.label, v, labels=CLASSES, average=None), 4).tolist())),
                   "confusion": confusion_matrix(df.label, v, labels=CLASSES).tolist()}
               for k, v in preds.items()}
    summary["scripts_agreement_rate"] = round(float(out.scripts_agree.mean()), 4)
    summary["protocol"] = {"folds_file": str(args.folds.name), "calendar_years": args.calendar_years,
                           "purge_pairs": bool(args.purge_pairs), "random_control": args.random_control, "n_folds": int(df.fold.nunique()),
                           "train_families_purged_total": int(fs[fs.variant == "ocr_both"].n_train_purged.sum())}
    if len(df) >= 50:
        rng = np.random.default_rng(42)
        g = df.group_id_v2.to_numpy()
        ug = np.unique(g)
        gi = {k: np.where(g == k)[0] for k in ug}
        y_ = df.label.to_numpy()
        p_ = preds["ocr_both"].astype(str)
        bs = [macro_f1(y_[ii], p_[ii]) for ii in (np.concatenate([gi[k] for k in rng.choice(ug, len(ug))]) for _ in range(2000))]
        summary["ocr_both"]["ci95_group_bootstrap"] = [round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)]
    (args.out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
