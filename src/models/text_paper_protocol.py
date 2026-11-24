"""Evaluation protocol for the text corpus (Tables 2 and 5 in the paper).

  A) Repeated, nested, stratified group cross-validation
     (R repetitions x K outer folds, groups = group_id_v2).
     Hyperparameters are selected by an INNER group CV on the outer training part only
     (nested CV) -> no optimistic selection bias.
     The folds are written to folds.csv so that the transformer and LLM scripts
     use exactly the SAME folds.
  B) Temporal hold-out: training on years <= 2019, testing on 2021-2026
     (the years 2020 and 2025 are not present in the corpus).
  C) Leave-one-level-out (municipal / district / state).

Statistics:
  - per model: mean and SD of macro-F1 over all R*K outer folds, plus a 95% bootstrap CI
    (group bootstrap of the pooled out-of-fold predictions of the first repetition, 2000 resamples);
  - paired comparisons with the reference model: corrected resampled t-test
    (Nadeau & Bengio, 2003) over per-fold differences, and McNemar's test on the
    out-of-fold predictions of the first repetition.

Command (from the repository root; default R=5, K=5):
    python src/models/text_paper_protocol.py
    python src/models/text_paper_protocol.py --repeats 1 --quick   # quick check

Output: reports/ml/text_v2/
    folds.csv, cv_fold_scores.csv, oof_predictions_rep0.csv,
    summary.json, table_text_cv.tex, temporal.json, leave_level_out.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
IN_PATH = REPO_ROOT / "data" / "interim" / "text_expanded" / "problems_ml_ready_v2.jsonl"
OUT_DIR = REPO_ROOT / "reports" / "ml" / "text_v2"
CLASSES = ["Algebra", "Geometrija", "Teorija brojeva", "Kombinatorika"]
SEED = 42
TEXT = "model_text"
GROUP = "group_id_v2"
REFERENCE_MODEL = "word_svm"  # reference model; all comparisons are made against it


# --------------------------------------------------------------------------
# Models and hyperparameter grids (the inner CV selects the combination)
# --------------------------------------------------------------------------
def word_vec(**kw):
    return TfidfVectorizer(sublinear_tf=True, max_df=0.95, **kw)


def char_vec(**kw):
    return TfidfVectorizer(analyzer="char_wb", sublinear_tf=True, **kw)


def model_space(quick: bool) -> dict[str, tuple[Pipeline, dict]]:
    Cs_svm = [0.06, 0.25, 1.0] if quick else [0.03, 0.06, 0.125, 0.25, 0.5, 1.0]
    Cs_lr = [1.0, 4.0] if quick else [0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
    return {
        "majority": (Pipeline([("v", word_vec()), ("clf", DummyClassifier(strategy="most_frequent"))]), {}),
        "word_lr": (
            Pipeline([("v", word_vec()), ("clf", LogisticRegression(max_iter=3000))]),
            {"v__ngram_range": [(1, 1), (1, 2)], "v__min_df": [1, 2], "clf__C": Cs_lr,
             "clf__class_weight": [None, "balanced"]},
        ),
        "word_svm": (
            Pipeline([("v", word_vec()), ("clf", LinearSVC(max_iter=10000))]),
            {"v__ngram_range": [(1, 1), (1, 2)], "v__min_df": [1, 2], "clf__C": Cs_svm,
             "clf__class_weight": [None, "balanced"]},
        ),
        "char_svm": (
            Pipeline([("v", char_vec(min_df=2)), ("clf", LinearSVC(max_iter=10000, class_weight="balanced"))]),
            {"v__ngram_range": [(2, 5), (3, 6)], "clf__C": Cs_svm},
        ),
        "word_char_svm": (
            Pipeline([
                ("v", FeatureUnion([("w", word_vec(ngram_range=(1, 2), min_df=2)),
                                    ("c", char_vec(ngram_range=(2, 5), min_df=2))])),
                ("clf", LinearSVC(max_iter=10000, class_weight="balanced")),
            ]),
            {"clf__C": Cs_svm},
        ),
        "complement_nb": (
            Pipeline([("v", word_vec(ngram_range=(1, 2), min_df=2)), ("clf", ComplementNB())]),
            {"clf__alpha": [0.1, 0.3, 1.0]},
        ),
    }


# --------------------------------------------------------------------------
def macro_f1(y, p):
    return f1_score(y, p, labels=CLASSES, average="macro", zero_division=0)


def inner_select(pipe, grid, X, y, g, seed, n_inner):
    """Inner group CV: returns the best combination by macro-F1."""
    if not grid:
        return {}, float("nan")
    keys = list(grid)
    combos = [dict(zip(keys, v)) for v in itertools.product(*(grid[k] for k in keys))]
    folds = list(StratifiedGroupKFold(n_inner, shuffle=True, random_state=seed).split(X, y, g))
    best, best_score = None, -1.0
    for params in combos:
        scores = []
        for a, b in folds:
            m = clone(pipe).set_params(**params).fit(X[a], y[a])
            scores.append(macro_f1(y[b], m.predict(X[b])))
        s = float(np.mean(scores))
        if s > best_score + 1e-12:  # ties keep the first (simpler) combination
            best, best_score = params, s
    return best, best_score


def fit_predict(name, space, Xtr, ytr, gtr, Xte, seed, n_inner):
    pipe, grid = space[name]
    params, inner = inner_select(pipe, grid, Xtr, ytr, gtr, seed, n_inner)
    model = clone(pipe).set_params(**params).fit(Xtr, ytr)
    return model.predict(Xte), params, inner


def group_bootstrap_ci(y, p, groups, n_boot=2000, seed=SEED):
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    idx_by_g = {gg: np.where(groups == gg)[0] for gg in uniq}
    vals = []
    for _ in range(n_boot):
        sample = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by_g[s] for s in sample])
        vals.append(macro_f1(y[idx], p[idx]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def corrected_resampled_ttest(diffs, n_train, n_test):
    """Nadeau & Bengio (2003) variance correction for repeated CV."""
    d = np.asarray(diffs, float)
    k = len(d)
    var = d.var(ddof=1)
    if var == 0:
        return float("nan"), float("nan")
    t = d.mean() / np.sqrt((1 / k + n_test / n_train) * var)
    p = 2 * stats.t.sf(abs(t), df=k - 1)
    return float(t), float(p)


def mcnemar(y, pa, pb):
    a_ok, b_ok = pa == y, pb == y
    n01, n10 = int(np.sum(a_ok & ~b_ok)), int(np.sum(~a_ok & b_ok))
    if n01 + n10 == 0:
        return n01, n10, 1.0
    p = stats.binomtest(min(n01, n10), n01 + n10, 0.5).pvalue  # exact McNemar test
    return n01, n10, float(p)


# --------------------------------------------------------------------------
def run_shift(df, X, y, g, names, space, args) -> tuple[dict, dict]:
    """B) temporal hold-out and C) leave-one-level-out. All training records whose group
    (exact / near-duplicate problems) has a member in the test part are removed, so that
    no problem appears on both sides of the split."""
    def purge(tr, te):
        test_groups = set(g[te])
        keep = np.array([gg not in test_groups for gg in g[tr]])
        return tr[keep], int((~keep).sum())

    temporal = {}
    tr, te = np.where(df.year <= 2019)[0], np.where(df.year >= 2021)[0]
    tr, n_purged = purge(tr, te)
    print(f"temporal: {n_purged} records removed from training (same problems as in test)")
    for n in names:
        pred, params, _ = fit_predict(n, space, X[tr], y[tr], g[tr], X[te], SEED, args.inner)
        lo, hi = group_bootstrap_ci(y[te], pred.astype(str), g[te])
        temporal[n] = {"n_train": len(tr), "n_test": len(te), "n_train_purged": n_purged,
                       "macro_f1": round(macro_f1(y[te], pred), 4),
                       "ci95": [round(lo, 4), round(hi, 4)], "accuracy": round(accuracy_score(y[te], pred), 4),
                       "params": json.dumps(params, default=str)}
    (args.out / "temporal.json").write_text(json.dumps(temporal, ensure_ascii=False, indent=2), encoding="utf-8")

    lolo = {}
    for lvl in sorted(df.nivo.unique()):
        tr, te = np.where(df.nivo != lvl)[0], np.where(df.nivo == lvl)[0]
        tr, n_purged = purge(tr, te)
        print(f"level {lvl}: {n_purged} records removed from training")
        lolo[lvl] = {}
        for n in names:
            pred, params, _ = fit_predict(n, space, X[tr], y[tr], g[tr], X[te], SEED, args.inner)
            lolo[lvl][n] = {"n_train": len(tr), "n_test": len(te), "n_train_purged": n_purged,
                            "macro_f1": round(macro_f1(y[te], pred), 4)}
    (args.out / "leave_level_out.json").write_text(json.dumps(lolo, ensure_ascii=False, indent=2), encoding="utf-8")
    print("temporal:", {k: v["macro_f1"] for k, v in temporal.items()})
    print("leave-one-level-out:", {lv: {k: v["macro_f1"] for k, v in d.items()} for lv, d in lolo.items()})
    return temporal, lolo


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=IN_PATH)
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--inner", type=int, default=3)
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--quick", action="store_true", help="smaller hyperparameter grids (only to check that everything runs)")
    ap.add_argument("--shift-only", action="store_true", help="only the temporal and leave-one-level-out checks (without the main CV)")
    args = ap.parse_args()

    with args.input.open(encoding="utf-8") as f:
        df = pd.DataFrame([json.loads(line) for line in f if line.strip()]).reset_index(drop=True)
    df["year"] = df["id"].str[:4].astype(int)
    X, y, g = df[TEXT].to_numpy(), df["oblast"].to_numpy(), df[GROUP].to_numpy()
    space = model_space(args.quick)
    names = args.models or list(space)
    args.out.mkdir(parents=True, exist_ok=True)
    if args.shift_only:
        run_shift(df, X, y, g, names, space, args)
        return 0

    # ---- A) repeated nested group CV -----------------------------------------
    fold_rows, folds_assign = [], []
    oof = {n: np.empty(len(df), dtype=object) for n in names}
    for r in range(args.repeats):
        sgkf = StratifiedGroupKFold(args.folds, shuffle=True, random_state=SEED + r)
        for k, (tr, te) in enumerate(sgkf.split(X, y, g)):
            assert not set(g[tr]) & set(g[te]), "group leakage!"
            folds_assign += [{"id": df["id"][i], "repeat": r, "fold": k} for i in te]
            for n in names:
                t0 = time.time()
                pred, params, inner = fit_predict(n, space, X[tr], y[tr], g[tr], X[te], SEED + r, args.inner)
                if r == 0:
                    oof[n][te] = pred
                fold_rows.append({
                    "model": n, "repeat": r, "fold": k, "n_train": len(tr), "n_test": len(te),
                    "macro_f1": macro_f1(y[te], pred), "accuracy": accuracy_score(y[te], pred),
                    "inner_cv_macro_f1": inner, "params": json.dumps(params, default=str),
                    "seconds": round(time.time() - t0, 1),
                })
                print(f"rep {r} fold {k} {n:15s} F1={fold_rows[-1]['macro_f1']:.4f} {params}")
    pd.DataFrame(folds_assign).to_csv(args.out / "folds.csv", index=False)
    scores = pd.DataFrame(fold_rows)
    scores.to_csv(args.out / "cv_fold_scores.csv", index=False)

    oof_df = df[["id", "oblast", "nivo", "razred", "kategorija", "year", GROUP]].copy()
    for n in names:
        oof_df[f"pred_{n}"] = oof[n]
    oof_df.to_csv(args.out / "oof_predictions_rep0.csv", index=False)

    summary = {"protocol": {"repeats": args.repeats, "outer_folds": args.folds, "inner_folds": args.inner,
                            "group_column": GROUP, "text_column": TEXT, "n_records": len(df),
                            "n_groups": int(pd.Series(g).nunique()), "quick_grid": args.quick},
               "models": {}}
    ref = scores[scores.model == REFERENCE_MODEL].sort_values(["repeat", "fold"]) if REFERENCE_MODEL in names else None
    for n in names:
        s = scores[scores.model == n].sort_values(["repeat", "fold"])
        lo, hi = group_bootstrap_ci(y, oof[n].astype(str), g)
        entry = {
            "macro_f1_mean": round(s.macro_f1.mean(), 4), "macro_f1_sd": round(s.macro_f1.std(ddof=1), 4),
            "accuracy_mean": round(s.accuracy.mean(), 4),
            "oof_rep0_macro_f1": round(macro_f1(y, oof[n].astype(str)), 4),
            "oof_rep0_ci95": [round(lo, 4), round(hi, 4)],
            "per_class_f1_rep0": dict(zip(CLASSES, np.round(f1_score(y, oof[n].astype(str), labels=CLASSES, average=None, zero_division=0), 4).tolist())),
            "confusion_rep0": confusion_matrix(y, oof[n].astype(str), labels=CLASSES).tolist(),
        }
        if ref is not None and n != REFERENCE_MODEL:
            diffs = s.macro_f1.to_numpy() - ref.macro_f1.to_numpy()
            t, p = corrected_resampled_ttest(diffs, s.n_train.mean(), s.n_test.mean())
            n01, n10, pm = mcnemar(y, oof[REFERENCE_MODEL].astype(str), oof[n].astype(str))
            entry["vs_reference"] = {"reference": REFERENCE_MODEL, "mean_diff": round(diffs.mean(), 4),
                                     "corrected_t": round(t, 3), "p_corrected_t": round(p, 4),
                                     "mcnemar_ref_only_correct": n01, "mcnemar_model_only_correct": n10,
                                     "p_mcnemar": round(pm, 4)}
        summary["models"][n] = entry
    (args.out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    # LaTeX table (for Overleaf: \input{tables/table_text_cv.tex})
    lines = ["\\begin{tabular}{lccc}", "\\toprule",
             "\\textbf{Model} & \\textbf{Macro-F1 (mean $\\pm$ SD)} & \\textbf{95\\% CI} & \\textbf{Accuracy} \\\\",
             "\\midrule"]
    for n in names:
        e = summary["models"][n]
        name_tex = n.replace("_", r"\_")
        lo_ci, hi_ci = e["oof_rep0_ci95"]
        lines.append(f"{name_tex} & {e['macro_f1_mean']:.3f} $\\pm$ {e['macro_f1_sd']:.3f} & "
                     f"[{lo_ci:.3f}, {hi_ci:.3f}] & {e['accuracy_mean']:.3f} " + r"\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (args.out / "table_text_cv.tex").write_text("\n".join(lines), encoding="utf-8")

    temporal, lolo = run_shift(df, X, y, g, names, space, args)

    print(json.dumps({k: {kk: v[kk] for kk in ("macro_f1_mean", "macro_f1_sd", "oof_rep0_ci95")}
                      for k, v in summary["models"].items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
