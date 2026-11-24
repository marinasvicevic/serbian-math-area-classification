"""All figures of the paper, numbered in order of appearance: FIGURE/Figure1.png, Figure2.png, ...

Run from the repository root:
    python src/tools/make_figures.py                 # all figures whose input data exist
    python src/tools/make_figures.py --only 3 5      # selected figures

Output: FIGURE/
    FigureN.png  (600 dpi)
    FigureN.pdf / FigureN.svg  (vector versions of charts and diagrams)

Figures:
  1  Example problem family (Cyrillic + Latin image, same label); needs the images   --family
  2  Distribution of areas: (a) text, (b) Kangaroo
  3  Study overview (vector diagram)
  4  Per-area F1 of the models (heat map): (a) text, (b) Kangaroo images
  5  Confusion matrices: (a) text, best model; (b) images, OCR route; (c) images, Claude (Latin script)
  6  Macro-F1 by age category (Kangaroo): OCR, Claude, Qwen and CNN
  7  Model agreement as a signal for expert review: accuracy of the consensus label by number of agreeing models

All text in the figures is in English; Serbian appears only in the problem images themselves.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
OUT = REPO_ROOT / "FIGURE"
DPI = 600

# ---- input files ----------------------------------------------------------------------------
TEXT_DATA = REPO_ROOT / "data" / "interim" / "text_expanded" / "problems_ml_ready_v2.jsonl"
TEXT_OOF = REPO_ROOT / "reports" / "ml" / "text_v2" / "oof_predictions_rep0.csv"
IMG_FOLDS = REPO_ROOT / "reports" / "image" / "v2" / "image_folds_v2.csv"
IMG_ANNOT = REPO_ROOT / "reports" / "image" / "final_image_annotations.csv"
OCR_OOF = REPO_ROOT / "reports" / "image" / "v2" / "ocr_text_cv" / "oof_predictions.csv"
V2 = REPO_ROOT / "reports" / "image" / "v2"
CLAUDE_IMG = V2 / "vlm_anthropic_claude-opus-5-5_{s}_kangaroo-img-v1" / "predictions.csv"
QWEN_IMG = V2 / "vlm_local_Qwen_Qwen3.6-27B_{s}_kangaroo-img-v1" / "predictions.csv"
CNN_OOF = V2 / "cnn_cv" / "resnet18_256x768" / "oof_family_probs.csv"
CNN512_OOF = V2 / "cnn_cv" / "resnet18_512x1536" / "oof_family_probs.csv"
CONVNEXT_OOF = V2 / "cnn_cv" / "convnext_tiny_384x1152" / "oof_family_probs.csv"
TV2 = REPO_ROOT / "reports" / "ml" / "text_v2"
BERTIC_PRED = TV2 / "transformer_classla_bcms-bertic" / "predictions.csv"
XLMR_PRED = TV2 / "transformer_xlm-roberta-base" / "predictions.csv"
CLAUDE_TXT = TV2 / "llm_anthropic_claude-opus-5-5_0shot_text-v2" / "predictions.csv"
QWEN_TXT = TV2 / "llm_openrouter_qwen_qwen3.6-27b_0shot_text-v2_host-Alibaba" / "predictions.csv"

# ---- English names ------------------------------------------------------------------------
EN = {"Algebra": "Algebra", "Geometrija": "Geometry", "Teorija brojeva": "Number theory",
      "Kombinatorika": "Combinatorics", "Logika": "Logic"}
TEXT_CLASSES = ["Algebra", "Geometrija", "Teorija brojeva", "Kombinatorika"]
IMG_CLASSES = ["Algebra", "Geometrija", "Teorija brojeva", "Logika"]
CATEGORIES = ["1", "2", "3-4", "5-6", "7-8", "9-10", "11-12"]
CAT_EN = {"1": "Grade 1", "2": "Grade 2", "3-4": "Grades\n3–4", "5-6": "Grades\n5–6", "7-8": "Grades\n7–8",
          "9-10": "Grades\n9–10", "11-12": "Grades\n11–12"}

# ---- colours (validated categorical palette; same order everywhere) ----------------------------
C_CLASS = {"Algebra": "#2a78d6", "Geometrija": "#eb6834", "Teorija brojeva": "#1baf7a",
           "Kombinatorika": "#eda100", "Logika": "#e87ba4"}
INK, INK2, MUTED, GRID = "#1b1b1a", "#52514e", "#8a8984", "#e4e3df"

# Kangaroo: in this competition the area "Teorija brojeva" is called "Brojevi" (Numbers). The internal key in
# the data stays "Teorija brojeva"; only the displayed name and the shade of green change for the image corpus.
C_NUMBERS = "#5e9e2e"


def nm(c: str, img: bool = False) -> str:
    """English area name; for the image corpus 'Teorija brojeva' -> 'Numbers'."""
    return "Numbers" if img and c == "Teorija brojeva" else EN[c]


def area_col(c: str, img: bool = False) -> str:
    return C_NUMBERS if img and c == "Teorija brojeva" else C_CLASS[c]

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 8.5,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42, "svg.fonttype": "none",
})
W1 = 5.2   # figure width in inches for \textwidth
WF = 6.8   # width for the full layout (\fulllength)


def save(fig, n: int, vector: bool = True):
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / f"Figure{n}.png", dpi=DPI, bbox_inches="tight", facecolor="white")
    if vector:
        fig.savefig(OUT / f"Figure{n}.pdf", bbox_inches="tight", facecolor="white")
        fig.savefig(OUT / f"Figure{n}.svg", bbox_inches="tight", facecolor="white")  # for Inkscape (text stays editable)
    plt.close(fig)
    print(f"  Figure{n}: OK")


def panel_label(ax, s, x=-0.02, y=1.04):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom", ha="right", color=INK)


def macro_f1(y, p, labels):
    return f1_score(y, p, labels=labels, average="macro", zero_division=0)


def group_ci(y, p, g, labels, n=1000, seed=42):
    rng = np.random.default_rng(seed)
    ug = np.unique(g)
    idx = {k: np.where(g == k)[0] for k in ug}
    v = []
    for _ in range(n):
        ii = np.concatenate([idx[k] for k in rng.choice(ug, len(ug))])
        v.append(macro_f1(y[ii], p[ii], labels))
    return np.percentile(v, [2.5, 97.5])


# =============================================================================================
# Figure 1: problem family
# =============================================================================================
def fig1(family: str):
    from PIL import Image

    ann = pd.read_csv(IMG_ANNOT, dtype=str)
    folds = pd.read_csv(IMG_FOLDS, dtype=str)
    row = ann[ann.task_family_id == family]
    if row.empty:
        raise SystemExit(f"Family {family} not found in {IMG_ANNOT.name}")
    row = row.iloc[0]
    lab = folds.set_index("task_family_id").label.get(family, row.get("label"))
    imgs = [Image.open(REPO_ROOT / row.cyrillic_image_path).convert("RGB"),
            Image.open(REPO_ROOT / row.latin_image_path).convert("RGB")]
    hs = [im.height / im.width for im in imgs]
    fig = plt.figure(figsize=(W1, W1 * 0.86 * sum(hs) + 0.55))
    gs = fig.add_gridspec(2, 1, height_ratios=hs, hspace=0.12, left=0.12, right=0.98, top=0.90, bottom=0.02)
    for k, (im, name) in enumerate(zip(imgs, ["Cyrillic", "Latin"])):
        ax = fig.add_subplot(gs[k])
        ax.imshow(im, interpolation="lanczos")
        ax.set_xticks([]), ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(True), s.set_color(GRID)
        ax.text(-0.015, 0.5, name, transform=ax.transAxes, rotation=90, ha="right", va="center",
                fontsize=9.5, color=INK, fontweight="bold")
    fig.text(0.12, 0.955, f"Kangaroo {row.year}, grades {str(row.category).replace('-', '–')}",
             fontsize=9, color=INK2, va="center", ha="left")
    fig.text(0.98, 0.955, f"Expert label: {nm(lab, img=True)}", fontsize=9, color="white", va="center", ha="right",
             fontweight="bold", bbox=dict(boxstyle="round,pad=0.3", fc=area_col(lab, img=True), ec="none"))
    save(fig, 1, vector=False)


# =============================================================================================
# Figure 2: distribution of areas
# =============================================================================================
def fig2():
    t = pd.read_json(TEXT_DATA, lines=True)
    i = pd.read_csv(IMG_FOLDS, dtype=str)
    fig, axes = plt.subplots(1, 2, figsize=(W1 + 0.6, 2.6), sharex=True)
    for ax, (data, classes, title, unit) in zip(axes, [
            (t.oblast, TEXT_CLASSES, "(a) Text corpus", "problems"),
            (i.label, IMG_CLASSES, "(b) Kangaroo image corpus", "problem families")]):
        cnt = data.value_counts().reindex(classes).fillna(0).astype(int)
        share = cnt / cnt.sum() * 100
        y = np.arange(len(classes))[::-1]
        ax.barh(y, share, height=0.62, color=[area_col(c, classes is IMG_CLASSES) for c in classes], edgecolor="white", linewidth=1.0)
        for yy, c in zip(y, classes):
            ax.text(share[c] + 1.0, yy, f"{share[c]:.1f}% ({cnt[c]:,})", va="center", fontsize=8.5, color=INK)
        ax.set_yticks(y, [nm(c, classes is IMG_CLASSES) for c in classes])
        ax.set_title(f"{title}\n(n = {cnt.sum():,} {unit})", loc="left", color=INK, pad=6, x=-0.42)
        ax.set_xlim(0, max(50, share.max() + 22))
        ax.xaxis.grid(True, color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_visible(False)
        ax.set_xlabel("Share of corpus (%)")
    fig.tight_layout(w_pad=2.0)
    save(fig, 2)
    return len(t), len(i)


# =============================================================================================
# Figure 3: study overview (diagram)
# =============================================================================================
def fig3(n_text: int | None, n_fam: int | None):
    """Diagram drawn directly in inches; smallest text 7.6 pt (about 7.4 pt at the final width in the paper)."""
    n_text = f"{n_text:,}" if n_text else "2,974"
    n_fam = f"{n_fam:,}" if n_fam else "2,822"
    n_img = f"{2 * int(n_fam.replace(',', '')):,}"
    Wd, Hd = 7.0, 5.35
    S_HEAD, S_TITLE, S_BODY, S_SUB, S_CHIP = 9.2, 8.8, 7.9, 7.6, 6.6
    fig = plt.figure(figsize=(Wd, Hd))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, Wd), ax.set_ylim(0, Hd), ax.axis("off")

    BLUE, BLUE_T, BLUE_E = "#2a78d6", "#eaf2fc", "#9cc1ee"
    ORNG, ORNG_T, ORNG_E = "#c4521f", "#fdf0ea", "#f2b79d"
    GREY_T = "#f4f4f2"

    def box(x, y, w, h, fc, ec, lw=0.9, r=0.06):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw))

    def txt(x, y, s, size=S_BODY, weight="normal", color=INK, ha="left", va="top", **kw):
        ax.text(x, y, s, fontsize=size, fontweight=weight, color=color, ha=ha, va=va, linespacing=1.3, **kw)

    def arrow(x0, y0, x1, y1, color=MUTED, lw=1.3):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=10,
                                     color=color, lw=lw, shrinkA=0, shrinkB=0))

    def chips(x, y_top, classes, max_w):
        """Area chips; the width is measured from the rendered text, wrapped into rows within max_w."""
        fig.canvas.draw()
        r = fig.canvas.get_renderer()
        inv = ax.transData.inverted()
        cx, cy, row_h = x, y_top, 0.24
        for c in classes:
            t = ax.text(0, 0, nm(c, classes is IMG_CLASSES), fontsize=S_CHIP, fontweight="bold", color="white", ha="center", va="center", zorder=3)
            bb = t.get_window_extent(renderer=r)
            tw = inv.transform((bb.x1, 0))[0] - inv.transform((bb.x0, 0))[0]
            cw = tw + 0.16
            if cx + cw > x + max_w:
                cx, cy = x, cy - row_h - 0.04
            ax.add_patch(FancyBboxPatch((cx, cy - row_h), cw, row_h, boxstyle="round,pad=0,rounding_size=0.06",
                                        fc=area_col(c, classes is IMG_CLASSES), ec="none", zorder=2))
            t.set_position((cx + cw / 2, cy - row_h / 2))
            cx += cw + 0.05

    cols = [(0.06, 1.70, "Corpora"), (1.98, 1.44, "Grouping"),
            (3.64, 1.94, "Models / routes"), (5.80, 1.14, "Evaluation")]
    for k, (x, w, name) in enumerate(cols, 1):
        ax.add_patch(plt.Circle((x + 0.11, Hd - 0.20), 0.105, color=INK))
        txt(x + 0.11, Hd - 0.20, str(k), size=8, weight="bold", color="white", ha="center", va="center")
        txt(x + 0.27, Hd - 0.20, name, size=S_HEAD, weight="bold", va="center")

    LH = 2.08                      # band height
    lanes = [(2.88, BLUE, BLUE_T, BLUE_E), (0.70, ORNG, ORNG_T, ORNG_E)]
    bh, gap = 0.62, 0.11           # model boxes
    pad = 0.11
    mx, mw = cols[2][0], cols[2][1]
    ex, ew = cols[3][0], cols[3][1]

    def model_boxes(y, edge, items, col, letters=None):
        for k, (a, b) in enumerate(items):
            yy = y + LH - (k + 1) * bh - k * gap
            box(mx, yy, mw, bh, "white", edge)
            x0 = mx + pad
            if letters:
                ax.add_patch(plt.Circle((mx + 0.19, yy + bh / 2), 0.11, color=col))
                txt(mx + 0.19, yy + bh / 2, letters[k], size=8, weight="bold", color="white", ha="center", va="center")
                x0 = mx + 0.38
            txt(x0, yy + bh - 0.09, a, size=S_TITLE - 0.4, weight="bold")
            txt(x0, yy + bh - 0.34, b, size=S_SUB, color=INK2)

    # ---- TEXT ----
    y, col, tint, edge = lanes[0]
    box(cols[0][0], y, cols[0][1], LH, tint, edge)
    txt(cols[0][0] + pad, y + LH - 0.11, "Text corpus", size=S_TITLE + 0.6, weight="bold", color=col)
    txt(cols[0][0] + pad, y + LH - 0.42,
        f"Serbian high-school\ncompetitions, 1999–2026\n{n_text} problems\nLaTeX, Latin script")
    chips(cols[0][0] + pad, y + 0.92, TEXT_CLASSES, cols[0][1] - 2 * pad)

    box(cols[1][0], y, cols[1][1], LH, "white", edge)
    txt(cols[1][0] + pad, y + LH - 0.11, "Unit: problem", size=S_TITLE, weight="bold", color=col)
    txt(cols[1][0] + pad, y + LH - 0.42,
        "Exact and near-\nduplicate problems\nform one group\n\nchar 4-gram TF-IDF,\ncosine ≥ 0.70,\nmanually reviewed")
    model_boxes(y, edge, [("TF-IDF (word / char)", "SVM · LR · Compl. NB"),
                          ("Fine-tuned encoders", "BERTić · XLM-R"),
                          ("Zero-shot LLMs", "Claude Opus 5.5 · Qwen3.6")], col)
    box(ex, y, ew, LH, "white", edge)
    txt(ex + pad, y + LH - 0.11, "Primary", size=S_TITLE, weight="bold", color=col)
    txt(ex + pad, y + LH - 0.40, "Nested 5×5\nstratified\ngroup CV")
    txt(ex + pad, y + LH - 1.05, "Robustness", size=S_TITLE, weight="bold", color=col)
    txt(ex + pad, y + LH - 1.34, "Temporal split\nLeave-one-\nlevel-out")

    # ---- IMAGES ----
    y, col, tint, edge = lanes[1]
    box(cols[0][0], y, cols[0][1], LH, tint, edge)
    txt(cols[0][0] + pad, y + LH - 0.11, "Kangaroo corpus", size=S_TITLE + 0.6, weight="bold", color=col)
    txt(cols[0][0] + pad, y + LH - 0.42,
        f"Kangaroo, 2014–2026\n{n_fam} problem families\nCyrillic + Latin image\n= {n_img} images")
    chips(cols[0][0] + pad, y + 0.92, IMG_CLASSES, cols[0][1] - 2 * pad)

    box(cols[1][0], y, cols[1][1], LH, "white", edge)
    txt(cols[1][0] + pad, y + LH - 0.11, "Unit: year", size=S_TITLE, weight="bold", color=col)
    txt(cols[1][0] + pad, y + LH - 0.42,
        "Shared problems of\nall grades together;\nproblems reused in\nlater years join the\nearliest year (OCR\n"
        "similarity ≥ 0.80,\nreviewed)")
    model_boxes(y, edge, [("Pixels → CNN", "ResNet-18 · ConvNeXt-T"),
                          ("OCR → text model", "Tesseract · TF-IDF + SVM"),
                          ("Image → VLM", "Claude Opus 5.5 · Qwen3.6")], col, letters="ABC")
    box(ex, y, ew, LH, "white", edge)
    txt(ex + pad, y + LH - 0.11, "Primary", size=S_TITLE, weight="bold", color=col)
    txt(ex + pad, y + LH - 0.40, "Leave-one-\nyear-out\n(13 folds),\nper family")
    txt(ex + pad, y + LH - 1.25, "Control", size=S_TITLE, weight="bold", color=col)
    txt(ex + pad, y + LH - 1.54, "Document-\nlevel groups")

    for (y, col, *_r) in lanes:
        ym = y + LH / 2
        for (x0, w0, _), (x1, _w, _n) in zip(cols[:-1], cols[1:]):
            arrow(x0 + w0 + 0.03, ym, x1 - 0.03, ym, color=col)

    box(0.06, 0.06, Wd - 0.12, 0.48, GREY_T, GRID)
    txt(0.20, 0.30, "Both corpora", size=S_TITLE, weight="bold", va="center")
    txt(1.42, 0.40, "macro-F1  ·  95% group-bootstrap CI  ·  same folds for all models of a corpus",
        size=S_SUB, color=INK2, va="center")
    txt(1.42, 0.20, "Tests:  text – corrected t-test, McNemar, Holm  ·  images – paired bootstrap of Δ macro-F1, McNemar",
        size=S_SUB, color=INK2, va="center")
    save(fig, 3)


# =============================================================================================
# Figure 4: per-area F1 (heat map)
# =============================================================================================
def _cnn_per_area(path: Path, y: pd.Series) -> np.ndarray:
    """Per-area F1 for each seed separately, then averaged over seeds (as in Table 3)."""
    c = pd.read_csv(path, dtype=str)
    return np.mean([_per_area(y, d.set_index("task_family_id").pred, IMG_CLASSES)
                    for _, d in c.groupby("seed")], axis=0)


def _per_area(y: pd.Series, p: pd.Series, classes) -> np.ndarray:
    p = p.reindex(y.index)
    return f1_score(y.values, p.fillna("__none__").values, labels=classes, average=None, zero_division=0)


def fig4():
    # (a) text: first repetition of the outer CV for trained models, same folds for zero-shot models
    r = pd.read_csv(TEXT_OOF, dtype=str).set_index("id")
    y = r.oblast

    def rep0(path):
        d = pd.read_csv(path, dtype=str)
        if "repeat" in d:
            d = d[d.repeat == "0"]
        return d.drop_duplicates("id", keep="last").set_index("id").pred

    T = [("Word SVM", r.pred_word_svm), ("Char SVM", r.pred_char_svm),
         ("Word+char SVM", r.pred_word_char_svm), ("BERTić", rep0(BERTIC_PRED)),
         ("XLM-R", rep0(XLMR_PRED)), ("Claude Opus 5.5", rep0(CLAUDE_TXT)), ("Qwen3.6-27B", rep0(QWEN_TXT))]
    MT = np.array([_per_area(y, p, TEXT_CLASSES) for _, p in T])

    # (b) images: Claude and Qwen = average of the Latin- and Cyrillic-script F1
    o = pd.read_csv(OCR_OOF, dtype=str).set_index("task_family_id")
    yi = o.label
    rows, script_gap = [], 0.0
    for name, path in [("ResNet-18 256×768", CNN_OOF), ("ResNet-18 512×1536", CNN512_OOF),
                       ("ConvNeXt-T 384×1152", CONVNEXT_OOF)]:
        rows.append((name, _cnn_per_area(path, yi)))
    rows.append(("OCR route", _per_area(yi, o.pred_ocr_both, IMG_CLASSES)))
    for name, tmpl in [("Claude Opus 5.5", CLAUDE_IMG), ("Qwen3.6-27B", QWEN_IMG)]:
        f = [_per_area(yi, pd.read_csv(Path(str(tmpl).format(s=sc)), dtype=str).set_index("id").pred, IMG_CLASSES)
             for sc in ("lat", "cyr")]
        script_gap = max(script_gap, float(np.abs(f[0] - f[1]).max()))
        rows.append((name, (f[0] + f[1]) / 2))
    MI = np.array([v for _, v in rows])

    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("blues", ["#f4f8fd", "#9cc3ee", "#2a78d6", "#14467f"])
    fig, axes = plt.subplots(1, 2, figsize=(WF + 0.4, 3.5), gridspec_kw={"wspace": 0.72})
    out = {}
    for ax, M, rnames, classes, title in [
            (axes[0], MT, [n for n, _ in T], TEXT_CLASSES, "(a) Text corpus"),
            (axes[1], MI, [n for n, _ in rows], IMG_CLASSES, "(b) Kangaroo image corpus")]:
        lo, hi = M.min(), M.max()
        ax.imshow(M, cmap=cmap, vmin=lo - 0.05 * (hi - lo), vmax=hi, aspect="auto")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                v = M[i, j]
                best = np.isclose(round(v, 3), round(M[:, j].max(), 3))
                ax.text(j, i, f"{v:.3f}".lstrip("0"), ha="center", va="center", fontsize=8.6,
                        color="white" if (v - lo) / (hi - lo) > 0.55 else INK, fontweight="bold" if best else "normal")
                if best:
                    ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fill=False, ec="#eb6834", lw=1.8))
        heads = [nm(c, classes is IMG_CLASSES).replace("Number theory", "Number\ntheory").replace("Combinatorics", "Combi-\nnatorics")
                 for c in classes]
        ax.set_xticks(range(len(classes)), heads, fontsize=7.4)
        ax.set_yticks(range(len(rnames)), rnames, fontsize=8.5)
        ax.xaxis.tick_top()
        ax.tick_params(length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title(title, loc="left", fontweight="bold", color=INK, pad=30)
        out[title] = {n: [round(float(v), 3) for v in M[i]] for i, n in enumerate(rnames)}
    save(fig, 4)
    print("   ", json.dumps(out, ensure_ascii=False))
    print(f"    largest Latin-Cyrillic per-area difference (Claude, Qwen): {script_gap:.3f}")


# =============================================================================================
# Figure 5: confusion matrices
# =============================================================================================
def confusion_panel(ax, y, p, classes, title, cbar=False):
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y, p, labels=classes)
    pct = cm / cm.sum(axis=1, keepdims=True) * 100
    im = ax.imshow(pct, cmap="Blues", vmin=0, vmax=100)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, f"{pct[i, j]:.0f}", ha="center", va="center", fontsize=9,
                    color="white" if pct[i, j] > 55 else INK, fontweight="bold" if i == j else "normal")
    short = {"Algebra": "Alg", "Geometrija": "Geo", "Teorija brojeva": "NT", "Kombinatorika": "Comb", "Logika": "Logic"}
    if classes is IMG_CLASSES:
        short = {**short, "Teorija brojeva": "Num"}
    ax.set_xticks(range(len(classes)), [short[c] for c in classes])
    ax.set_yticks(range(len(classes)), [short[c] for c in classes])
    ax.set_xlabel("Predicted")
    ax.set_title(title, color=INK, fontsize=9.5, pad=6)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    mf = macro_f1(np.asarray(y), np.asarray(p), classes)
    ax.text(0.5, -0.36, f"macro-F1 = {mf:.3f}", transform=ax.transAxes, ha="center", fontsize=9, color=INK2)
    return im


def fig5():
    panels = []
    t = pd.read_csv(TEXT_OOF)
    panels.append((t.oblast, t.pred_word_char_svm, TEXT_CLASSES, "(a) Text: word+char\nTF-IDF + SVM"))
    o = pd.read_csv(OCR_OOF, dtype=str)
    panels.append((o.label, o.pred_ocr_both, IMG_CLASSES, "(b) Images: OCR →\nword/char SVM"))
    cl = Path(str(CLAUDE_IMG).format(s="lat"))
    if cl.exists():
        c = pd.read_csv(cl, dtype=str)
        panels.append((c["true"], c["pred"], IMG_CLASSES, "(c) Images: Claude\nOpus 5.5, zero-shot"))
    fig, axes = plt.subplots(1, len(panels), figsize=(WF * len(panels) / 3 + 0.6, 2.9), gridspec_kw={"wspace": 0.45})
    axes = np.atleast_1d(axes)
    for k, (ax, (y, p, cls, title)) in enumerate(zip(axes, panels)):
        im = confusion_panel(ax, y, p, cls, title)
        if k == 0:
            ax.set_ylabel("True")
    cb = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    cb.set_label("% of true area (row)", fontsize=9)
    cb.ax.tick_params(labelsize=8.5)
    cb.outline.set_visible(False)
    save(fig, 5)


# =============================================================================================
# Figure 6: macro-F1 by age category
# =============================================================================================
def fig6():
    """Macro-F1 by age category. For the CNN: macro-F1 averaged over 3 seeds (as in Table 3), not an ensemble."""
    folds = pd.read_csv(IMG_FOLDS, dtype=str).set_index("task_family_id")
    series = []  # (name, DataFrame with columns id, y, p0[, p1, p2])
    o = pd.read_csv(OCR_OOF, dtype=str)
    series.append(("OCR → SVM (trained)", pd.DataFrame({"id": o.task_family_id, "y": o.label, "p0": o.pred_ocr_both})))
    if CNN_OOF.exists():
        c = pd.read_csv(CNN_OOF, dtype=str)
        w = c.pivot(index="task_family_id", columns="seed", values="pred")
        w.columns = [f"p{i}" for i in range(w.shape[1])]
        w = w.reset_index().rename(columns={"task_family_id": "id"})
        w["y"] = folds.loc[w.id, "label"].values
        series.append(("ResNet-18 (trained, 3 seeds)", w))
    # zero-shot models: the two image versions (Latin and Cyrillic) are two runs -> average macro-F1, like CNN seeds
    for name, path in [("Claude Opus 5.5 (zero-shot)", CLAUDE_IMG), ("Qwen3.6-27B (zero-shot)", QWEN_IMG)]:
        parts = []
        for i, sc in enumerate(("lat", "cyr")):
            pth = Path(str(path).format(s=sc))
            if pth.exists():
                d = pd.read_csv(pth, dtype=str)
                parts.append(d[["id", "true", "pred"]].rename(columns={"pred": f"p{i}"}))
        if parts:
            m = parts[0]
            for q in parts[1:]:
                m = m.merge(q.drop(columns="true"), on="id")
            series.append((name, m.rename(columns={"true": "y"})))
    style = {"OCR": ("#2a78d6", "o"), "Claude": ("#eb6834", "s"), "ResNet": ("#1baf7a", "D"), "Qwen": ("#6250d6", "^")}
    fig, ax = plt.subplots(figsize=(W1 + 0.6, 3.0))
    xs = np.arange(len(CATEGORIES))
    ncat = None
    rng = np.random.default_rng(42)
    for name, df in series:
        col, mk = next(v for key, v in style.items() if name.startswith(key))
        pcols = [col for col in df.columns if col.startswith("p") and col[1:].isdigit()]
        df = df.copy()
        df["cat"] = folds.loc[df.id, "category"].values
        df["g"] = folds.loc[df.id, "group_id_v2"].values
        vals, lo, hi = [], [], []
        for cat in CATEGORIES:
            d = df[df.cat == cat]
            yy, gg = d.y.to_numpy(), d.g.to_numpy()
            P = [d[pc].to_numpy() for pc in pcols]
            v = float(np.mean([macro_f1(yy, pp, IMG_CLASSES) for pp in P]))
            ug = np.unique(gg)
            gi = {u: np.where(gg == u)[0] for u in ug}
            bs = []
            for _ in range(500):
                ii = np.concatenate([gi[u] for u in rng.choice(ug, len(ug))])
                bs.append(np.mean([macro_f1(yy[ii], pp[ii], IMG_CLASSES) for pp in P]))
            ci = np.percentile(bs, [2.5, 97.5])
            vals.append(v), lo.append(v - ci[0]), hi.append(ci[1] - v)
        ncat = df.cat.value_counts().reindex(CATEGORIES)
        v = np.array(vals)
        ax.fill_between(xs, v - np.array(lo), v + np.array(hi), color=col, alpha=0.13, lw=0)
        ax.plot(xs, v, color=col, marker=mk, ms=5.5, lw=1.8, label=name, mec="white", mew=0.6)
    ax.set_xticks(xs, [c.replace("-", "–") for c in CATEGORIES])
    ax.set_xlabel("Kangaroo category (school grade)")
    ax.set_ylabel("Macro-F1")
    ax.set_ylim(0.3, 0.95)
    ax.yaxis.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", length=0)
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    save(fig, 6)


# =============================================================================================
# Figure 7: model agreement as a signal for expert review
# =============================================================================================
def consensus_table(P: pd.DataFrame, y: pd.Series, ref: str):
    """For every item: how many models give the most frequent label, and that label
    (ties are resolved by the model passed as `ref`: word+char SVM for text, OCR route for images)."""
    rows = []
    for i, r in P.iterrows():
        vc = r.value_counts()
        top = int(vc.iloc[0])
        lab = vc.index[0] if (len(vc) == 1 or vc.iloc[0] > vc.iloc[1]) else r[ref]
        rows.append((top, lab))
    t = pd.DataFrame(rows, index=P.index, columns=["top", "lab"])
    t["ok"] = t.lab.values == y.loc[P.index].values
    return t


def fig7():
    # text: one model from each model family
    r = pd.read_csv(TEXT_OOF, dtype=str).set_index("id")
    b = pd.read_csv(BERTIC_PRED, dtype=str)
    b = b[b.repeat == "0"].drop_duplicates("id", keep="last").set_index("id").pred
    ct = pd.read_csv(CLAUDE_TXT, dtype=str).set_index("id").pred
    qt = pd.read_csv(QWEN_TXT, dtype=str).set_index("id").pred
    PT = pd.DataFrame({"svm": r.pred_word_char_svm, "bertic": b, "claude": ct, "qwen": qt}).loc[r.index]
    tt = consensus_table(PT, r.oblast, "svm")
    # images: OCR (both scripts), ResNet-18 (probabilities averaged over 3 seeds), Claude and Qwen (Latin script)
    o = pd.read_csv(OCR_OOF, dtype=str).set_index("task_family_id")
    c = pd.read_csv(CNN_OOF, dtype=str)
    pc = [f"p_{k}" for k in IMG_CLASSES]
    c[pc] = c[pc].astype(float)
    m = c.groupby("task_family_id")[pc].mean()
    cnn = pd.Series(np.array(IMG_CLASSES)[m.values.argmax(1)], index=m.index)
    cl = pd.read_csv(Path(str(CLAUDE_IMG).format(s="lat")), dtype=str).set_index("id").pred
    qw = pd.read_csv(Path(str(QWEN_IMG).format(s="lat")), dtype=str).set_index("id").pred
    PI = pd.DataFrame({"ocr": o.pred_ocr_both, "cnn": cnn, "claude": cl, "qwen": qw}).loc[o.index]
    ti = consensus_table(PI, o.label, "ocr")

    shades = ["#1f5fa8", "#2a78d6", "#7fb0ea"]
    labs = ["4 of 4", "3 of 4", "2 of 4 or fewer"]
    fig, axes = plt.subplots(1, 2, figsize=(W1 + 0.9, 2.75), sharey=True, gridspec_kw={"wspace": 0.18})
    summary = {}
    for ax, t, title, sub in [(axes[0], tt, "(a) Text corpus", "word+char SVM · BERTić · Claude · Qwen"),
                              (axes[1], ti, "(b) Kangaroo image corpus", "OCR route · ResNet-18 · Claude · Qwen")]:
        groups = [t.top == 4, t.top == 3, t.top <= 2]
        acc = [t.ok[g].mean() for g in groups]
        share = [g.mean() for g in groups]
        x = np.arange(3)
        ax.bar(x, acc, width=0.8, color=shades, edgecolor="white")
        for xi, a, sh in zip(x, acc, share):
            ax.text(xi, a + 0.02, f"{a * 100:.0f}%", ha="center", fontsize=9.5, fontweight="bold", color=INK)
            ax.text(xi, 0.05, f"{sh * 100:.0f}% of\nproblems", ha="center", fontsize=7.6, color="white", linespacing=1.15)
        ax.set_xticks(x, labs)
        ax.set_xlabel("Models agreeing on the area")
        ax.set_ylim(0, 1.05)
        ax.set_title(title, loc="left", color=INK, pad=20, fontweight="bold")
        ax.text(0, 1.035, sub, transform=ax.transAxes, fontsize=8.3, color=INK2, va="bottom")
        ax.yaxis.grid(True, color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        ax.tick_params(axis="x", length=0)
        summary[title] = {"share": [round(v, 3) for v in share], "accuracy": [round(v, 3) for v in acc],
                                        "consensus_accuracy_all": round(float(t.ok.mean()), 3)}
    axes[0].set_ylabel("Accuracy of the consensus label")
    save(fig, 7)
    print("   ", json.dumps(summary, ensure_ascii=False))


# =============================================================================================
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", type=int, nargs="*", default=None)
    ap.add_argument("--family", default="kgfam-2014-regular-9-10-t06", help="family shown in Figure 1")
    args = ap.parse_args()
    todo = set(args.only or range(1, 8))
    print(f"Output: {OUT}")
    n_text = n_fam = None
    jobs = {1: lambda: fig1(args.family), 4: fig4, 5: fig5, 6: fig6, 7: fig7}
    for n in sorted(todo):
        try:
            if n == 2:
                n_text, n_fam = fig2()
            elif n == 3:
                if n_text is None:
                    try:
                        n_text = len(pd.read_json(TEXT_DATA, lines=True))
                        n_fam = len(pd.read_csv(IMG_FOLDS))
                    except Exception:
                        pass
                fig3(n_text, n_fam)
            else:
                jobs[n]()
        except (FileNotFoundError, SystemExit) as e:
            print(f"  Figure{n}: skipped ({e})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
