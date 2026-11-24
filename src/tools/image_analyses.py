"""Additional analyses of the Kangaroo image results (Tables 3 and 6, Section 5.2 and 5.4 of the paper).

Computes, from the saved out-of-fold predictions only (no images or GPU needed):
  1. group-bootstrap 95% CIs of pooled macro-F1 for the CNN runs (for three seeds the pooled
     macro-F1 is averaged over seeds inside every bootstrap sample);
  2. paired group-bootstrap differences that involve the CNN (seed-averaged where applicable);
  3. agreement between the Latin- and Cyrillic-script predictions of every route, with the
     accuracy of the individual predictions on agreeing and disagreeing families;
  4. how often Claude/Qwen predict "Numbers" for problems labelled Numbers, per age category.

Conventions (as in the paper): macro-F1 over the four areas; an invalid answer is a false negative
for the true area only; CIs: 2000 resamples of groups, percentile interval, seed 42; paired
differences: 5000 resamples, seed 42. Groups: competition years (image_folds_v2.csv) or documents
(image_folds_doc.csv).

Command (from the repository root):
    python src/tools/image_analyses.py
Output: console
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
V2 = REPO_ROOT / "reports" / "image" / "v2"
CLASSES = ["Algebra", "Geometrija", "Teorija brojeva", "Logika"]
K = {c: i for i, c in enumerate(CLASSES)}


def load_folds(name: str) -> pd.DataFrame:
    return pd.read_csv(V2 / name).set_index("task_family_id")


def cnn_preds(run: str, ids: pd.Index, col: str = "pred") -> list[pd.Series]:
    """One prediction Series per seed for a CNN run (column pred, pred_lat or pred_cyr)."""
    d = pd.read_csv(V2 / "cnn_cv" / run / "oof_family_probs.csv")
    return [g.set_index("task_family_id")[col].reindex(ids) for _, g in d.groupby("seed")]


def vlm_pred(model_dir: str, ids: pd.Index) -> pd.Series:
    d = pd.read_csv(V2 / model_dir / "predictions.csv").drop_duplicates("id", keep="last")
    return d.set_index("id")["pred"].reindex(ids).fillna("INVALID")


def macro_f1_cm(cm: np.ndarray) -> float:
    """cm is 4x5: column 4 = invalid answer (false negative only)."""
    tp = np.diag(cm[:, :4]).astype(float)
    p, t = cm[:, :4].sum(0), cm.sum(1)
    return float(np.divide(2 * tp, p + t, out=np.zeros(4), where=(p + t) > 0).mean())


def group_cms(pred: pd.Series, y: np.ndarray, g: np.ndarray, ug: np.ndarray) -> np.ndarray:
    p = pred.map(K).fillna(4).astype(int).to_numpy()
    out = np.zeros((len(ug), 4, 5), dtype=int)
    for i, gg in enumerate(ug):
        m = g == gg
        np.add.at(out[i], (y[m], p[m]), 1)
    return out


def ci(preds: list[pd.Series], folds: pd.DataFrame, n: int = 2000, seed: int = 42) -> tuple[float, float, float]:
    """Pooled macro-F1 (averaged over seeds) and its group-bootstrap 95% CI."""
    y, g = folds.label.to_numpy(), folds.group_id_v2.to_numpy()
    ug = np.unique(g)
    idx = {x: np.where(g == x)[0] for x in ug}
    P = [p.to_numpy() for p in preds]
    f = lambda yy, pp: f1_score(yy, pp, labels=CLASSES, average="macro", zero_division=0)  # noqa: E731
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        s = rng.choice(ug, len(ug), replace=True)
        i = np.concatenate([idx[x] for x in s])
        vals.append(np.mean([f(y[i], p[i]) for p in P]))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(np.mean([f(y, p) for p in P])), float(lo), float(hi)


def delta(a: list[pd.Series], b: list[pd.Series], folds: pd.DataFrame, n: int = 5000, seed: int = 42):
    """Paired group bootstrap of macro-F1(A) - macro-F1(B); each side averaged over its seeds."""
    y = folds.label.map(K).to_numpy()
    g = folds.group_id_v2.to_numpy()
    ug = np.unique(g)
    A = [group_cms(p, y, g, ug) for p in a]
    B = [group_cms(p, y, g, ug) for p in b]
    f = lambda L, s: np.mean([macro_f1_cm(c[s].sum(0)) for c in L])  # noqa: E731
    allg = np.arange(len(ug))
    d0 = f(A, allg) - f(B, allg)
    rng = np.random.default_rng(seed)
    D = [f(A, s) - f(B, s) for s in (rng.integers(0, len(ug), len(ug)) for _ in range(n))]
    lo, hi = np.percentile(D, [2.5, 97.5])
    return float(d0), float(lo), float(hi)


def fmt(x):
    return f"{x[0]:+.3f} [{x[1]:+.3f}, {x[2]:+.3f}]"


def main() -> int:
    yf = load_folds("image_folds_v2.csv")
    df = load_folds("image_folds_doc.csv").reindex(yf.index)
    ids = yf.index
    base = cnn_preds("resnet18_256x768", ids)
    ocr = pd.read_csv(V2 / "ocr_text_cv" / "oof_predictions.csv").set_index("task_family_id").reindex(ids)

    print("1) CNN: pooled macro-F1 and 95% CI")
    for name, preds, folds in [
        ("ResNet-18 256x768, 3 seeds (year folds)", base, yf),
        ("ResNet-18 256x768, seed 0 (year folds)", base[:1], yf),
        ("ResNet-18 512x1536, seed 0 (year folds)", cnn_preds("resnet18_512x1536", ids), yf),
        ("ConvNeXt-T 384x1152, seed 0 (year folds)", cnn_preds("convnext_tiny_384x1152", ids), yf),
        ("ResNet-18 256x768, seed 0 (document folds)", cnn_preds("resnet18_256x768_docsplit", ids), df),
    ]:
        m, lo, hi = ci(preds, folds)
        print(f"   {name:45s} {m:.3f} [{lo:.3f}, {hi:.3f}]")

    print("2) Paired differences in pooled macro-F1 (year groups)")
    hi_res = cnn_preds("resnet18_512x1536", ids)
    convnext = cnn_preds("convnext_tiny_384x1152", ids)
    docsplit = cnn_preds("resnet18_256x768_docsplit", ids)
    for name, a, b in [
        ("ResNet-18 (3 seeds) - OCR route", base, [ocr.pred_ocr_both]),
        ("ResNet-18 512x1536 - baseline, seed 0", hi_res, base[:1]),
        ("ConvNeXt-T - baseline, seed 0", convnext, base[:1]),
        ("ResNet-18 512x1536 - OCR route", hi_res, [ocr.pred_ocr_both]),
        ("ConvNeXt-T - OCR route", convnext, [ocr.pred_ocr_both]),
        ("ResNet-18 seed 0: document split - year split", docsplit, base[:1]),
        ("ResNet-18 (3 seeds): Latin - Cyrillic", cnn_preds("resnet18_256x768", ids, "pred_lat"),
         cnn_preds("resnet18_256x768", ids, "pred_cyr")),
    ]:
        print(f"   {name:45s} {fmt(delta(a, b, yf))}")

    print("3) Agreement between the Latin and Cyrillic predictions")
    y = yf.label
    routes = {"OCR route": [(ocr.pred_ocr_lat, ocr.pred_ocr_cyr)],
              "Claude Opus 5.5": [(vlm_pred("vlm_anthropic_claude-opus-5-5_lat_kangaroo-img-v1", ids),
                                   vlm_pred("vlm_anthropic_claude-opus-5-5_cyr_kangaroo-img-v1", ids))],
              "Qwen3.6-27B": [(vlm_pred("vlm_local_Qwen_Qwen3.6-27B_lat_kangaroo-img-v1", ids),
                               vlm_pred("vlm_local_Qwen_Qwen3.6-27B_cyr_kangaroo-img-v1", ids))],
              "ResNet-18 (mean of 3 seeds)": list(zip(cnn_preds("resnet18_256x768", ids, "pred_lat"),
                                                      cnn_preds("resnet18_256x768", ids, "pred_cyr")))}
    for name, pairs in routes.items():
        agree = np.mean([(a == b).mean() for a, b in pairs])
        n_dis = np.mean([(a != b).sum() for a, b in pairs])
        acc_lat = np.mean([(a[a != b] == y[a != b]).mean() for a, b in pairs])
        acc_cyr = np.mean([(b[a != b] == y[a != b]).mean() for a, b in pairs])
        acc_agree = np.mean([(a[a == b] == y[a == b]).mean() for a, b in pairs])
        print(f"   {name:28s} agreement {100 * agree:.1f}% | disagreeing families {n_dis:.0f}: "
              f"accuracy Latin {100 * acc_lat:.1f}%, Cyrillic {100 * acc_cyr:.1f}% | agreeing: {100 * acc_agree:.1f}%")

    print("4) Problems labelled Numbers that are predicted as Numbers, per age category")
    cats = ["1", "2", "3-4", "5-6", "7-8", "9-10", "11-12"]
    cat = yf.category.astype(str)
    for name, d in [("Claude, Latin", "vlm_anthropic_claude-opus-5-5_lat_kangaroo-img-v1"),
                    ("Claude, Cyrillic", "vlm_anthropic_claude-opus-5-5_cyr_kangaroo-img-v1"),
                    ("Qwen, Latin", "vlm_local_Qwen_Qwen3.6-27B_lat_kangaroo-img-v1"),
                    ("Qwen, Cyrillic", "vlm_local_Qwen_Qwen3.6-27B_cyr_kangaroo-img-v1")]:
        p = vlm_pred(d, ids)
        cells = []
        for c in cats:
            m = (cat == c) & (y == "Teorija brojeva")
            cells.append(f"{c}: {int((p[m] == 'Teorija brojeva').sum())}/{int(m.sum())}")
        print(f"   {name:18s} " + "  ".join(cells))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
