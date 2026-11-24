"""Leakage-aware folds for the Kangaroo image corpus (primary protocol: leave-one-year-out).

Problem
-------
Grouping by `document_group_id` (= year/round/category) is not sufficient: in the Kangaroo
competition ADJACENT age categories of the same year often share problems (e.g. 3-4 and 5-6),
and a problem is occasionally reused in another year. Such a problem appears in the corpus as
TWO different families from two different documents -> it can end up in both training and test.

Solution
--------
1. Group = whole competition year (all categories and both rounds) -> problems shared within
   a year never straddle the split.
2. OCR text (ocr_kengur.py) is used to find candidate same-problem pairs
   (cosine over character 4-grams >= --threshold). Excel file for manual review:
   cross_family_pairs_review.xlsx (DA/NE = yes/no, plus konacna_oznaka = final label for pairs
   with different labels). After the review: --reviewed.
3. A problem reused from an earlier year (together with its same-year copies) is moved to the
   group of the earliest year in which it appears - whole years are NOT merged.
4. Folds: leave-one-year-out (13 folds).

Command:
    python src/data_image/build_image_folds_v2.py                       # candidate pairs + folds
    python src/data_image/build_image_folds_v2.py --reviewed reports/image/v2/cross_family_pairs_review.xlsx

Output: reports/image/v2/
    cross_family_pairs.csv   (all candidate pairs, including same-year pairs)
    image_folds_v2.csv       (task_family_id, label, year, group_id_v2, fold)
    image_folds_v2_report.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse.csgraph import connected_components
from scipy.sparse import coo_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
ANNOT = REPO_ROOT / "reports" / "image" / "final_image_annotations.csv"
OCR = REPO_ROOT / "data" / "image_interim" / "ocr" / "kengur_ocr_v1.jsonl"
OUT_DIR = REPO_ROOT / "reports" / "image" / "v2"


def write_review_excel(pairs: pd.DataFrame, fam: pd.DataFrame, path: Path) -> None:
    """Excel file for manual review: links to the images, OCR text and a DA/NE (yes/no) drop-down."""
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.worksheet.datavalidation import DataValidation

    if path.exists():
        print(f"NOTE: {path.name} already exists and was NOT overwritten (to keep the manual review).")
        return
    ocr = fam.set_index("task_family_id")["ocr_latin"].fillna("")
    wb = Workbook()
    for title, part in (("Medjugodisnji", pairs[~pairs.same_year]), ("Ista_godina", pairs[pairs.same_year])):
        ws = wb.active if title == "Medjugodisnji" else wb.create_sheet(title)
        ws.title = title
        cols = ["rb", "similarity", "family_a", "year_a", "category_a", "label_a", "slika_a", "ocr_a",
                "family_b", "year_b", "category_b", "label_b", "slika_b", "ocr_b", "razlicite_oznake", "isti_zadatak", "konacna_oznaka", "komentar"]
        ws.append(cols)
        for i, r in enumerate(part.itertuples(index=False), 1):
            ws.append([i, r.similarity, r.family_a, r.year_a, r.category_a, r.label_a, "otvori sliku",
                       ocr[r.family_a][:600], r.family_b, r.year_b, r.category_b, r.label_b, "otvori sliku",
                       ocr[r.family_b][:600], "DA" if r.label_a != r.label_b else "", "", "", ""])
            ws.cell(i + 1, 7).hyperlink = str((REPO_ROOT / r.image_a).resolve())
            ws.cell(i + 1, 13).hyperlink = str((REPO_ROOT / r.image_b).resolve())
            for c in (7, 13):
                ws.cell(i + 1, c).font = Font(color="0563C1", underline="single")
            if r.label_a != r.label_b:
                for c in range(1, 16):
                    ws.cell(i + 1, c).fill = PatternFill("solid", fgColor="FCE4D6")
            ws.cell(i + 1, 16).fill = PatternFill("solid", fgColor="FFF2CC")
            if r.label_a != r.label_b:
                ws.cell(i + 1, 17).fill = PatternFill("solid", fgColor="FFF2CC")
        for c in ws[1]:
            c.font = Font(bold=True)
            c.fill = PatternFill("solid", fgColor="DDEBF7")
        widths = [5, 9, 26, 7, 9, 14, 12, 55, 26, 7, 9, 14, 12, 55, 10, 12, 16, 25]
        for j, w in enumerate(widths):
            ws.column_dimensions[chr(65 + j)].width = w
        for row in ws.iter_rows(min_row=2):
            for c in row:
                c.alignment = Alignment(wrap_text=True, vertical="top")
        dv = DataValidation(type="list", formula1='"DA,NE"', allow_blank=True)
        ws.add_data_validation(dv)
        dv.add(f"P2:P{max(2, ws.max_row)}")
        dv2 = DataValidation(type="list", formula1='"Algebra,Geometrija,Teorija brojeva,Logika"', allow_blank=True)
        ws.add_data_validation(dv2)
        dv2.add(f"Q2:Q{max(2, ws.max_row)}")
        ws.freeze_panes = "C2"
    u = wb.create_sheet("Uputstvo")
    for line in [
        "Pairs of Kangaroo problems with very similar OCR text (candidates for the SAME problem).",
        "",
        "Sheet Medjugodisnji (REQUIRED): pairs from DIFFERENT years. In column isti_zadatak choose DA (yes) or NE (no).",
        "  DA = same problem (same text, possibly different numbers/answer order). NE = different problems.",
        "  Click 'otvori sliku' to open the image. Problems linked with DA are placed in the same group (same fold).",
        "",
        "Sheet Ista_godina (OPTIONAL): same-year pairs from different categories. They do not change the folds",
        "  (a whole year is already one group), but give the EXACT number of problems shared across categories,",
        "  i.e. the amount of leakage under a document-level split. A subset may be reviewed.",
        "",
        "Orange rows: the pair has DIFFERENT expert labels. If isti_zadatak = DA, choose one correct area in",
        "  column konacna_oznaka - it applies to both problems of the pair (as for the text corpus).",
        "  Orange rows must also be reviewed on sheet Ista_godina; other rows there are optional.",
        "If in doubt: DA.",
        "Save under the same name, close Excel, then run the script with --reviewed.",
    ]:
        u.append([line])
    u.column_dimensions["A"].width = 110
    wb.save(path)
    print(f"Review file created: {path}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotations", type=Path, default=ANNOT)
    ap.add_argument("--ocr", type=Path, default=OCR)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--threshold", type=float, default=0.80)
    ap.add_argument("--reviewed", type=Path, default=None,
                    help="cross_family_pairs_review.xlsx after manual review (column isti_zadatak = DA/NE, i.e. yes/no)")
    args = ap.parse_args()

    fam = pd.read_csv(args.annotations)
    ocr = pd.read_json(args.ocr, lines=True)
    fam = fam.merge(ocr[["task_family_id", "ocr_latin", "ocr_cyrillic_translit"]], on="task_family_id", how="left")
    if fam["ocr_latin"].isna().any():
        raise SystemExit("OCR is missing for some families - run ocr_kengur.py on the whole corpus first.")
    text = (fam["ocr_latin"].fillna("") + " " + fam["ocr_cyrillic_translit"].fillna("")).str.lower()

    X = TfidfVectorizer(analyzer="char_wb", ngram_range=(4, 4), sublinear_tf=True, min_df=2).fit_transform(text)
    S = (X @ X.T).tocoo()
    m = (S.row < S.col) & (S.data >= args.threshold)
    a, b, s = S.row[m], S.col[m], S.data[m]
    pairs = pd.DataFrame({
        "similarity": np.round(s, 4),
        "family_a": fam.task_family_id.values[a], "family_b": fam.task_family_id.values[b],
        "year_a": fam.year.values[a], "year_b": fam.year.values[b],
        "category_a": fam.category.astype(str).values[a], "category_b": fam.category.astype(str).values[b],
        "label_a": fam.label.values[a], "label_b": fam.label.values[b],
        "same_year": fam.year.values[a] == fam.year.values[b],
        "image_a": fam.latin_image_path.values[a], "image_b": fam.latin_image_path.values[b],
        "same_task": "",
    }).sort_values("similarity", ascending=False)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(args.out_dir / "cross_family_pairs.csv", index=False, encoding="utf-8")

    review_xlsx = args.out_dir / "cross_family_pairs_review.xlsx"
    review_info: dict = {}
    fam["label_original"] = fam["label"]
    if args.reviewed:
        sheets = pd.ExcelFile(args.reviewed).sheet_names
        parts = []
        for sh in ("Medjugodisnji", "Ista_godina"):
            if sh in sheets:
                d = pd.read_excel(args.reviewed, sheet_name=sh, dtype=str)
                d["odluka"] = d["isti_zadatak"].fillna("").astype(str).str.strip().str.upper()
                d["konacna"] = d["konacna_oznaka"].fillna("").astype(str).str.strip() if "konacna_oznaka" in d else ""
                d["list"] = sh
                d["excel_red"] = d.index + 2  # Excel row number (row 1 is the header)
                parts.append(d)
        rev = pd.concat(parts, ignore_index=True)
        cy = rev[rev["list"] == "Medjugodisnji"]
        n_empty = int((cy["odluka"] == "").sum())
        if n_empty:
            print(f"WARNING: {n_empty} cross-year pairs without a decision - treated as DA.")
        # same problem = DA, or (conservatively) no decision; NE pairs are discarded
        same = rev[rev["odluka"] != "NE"]
        # label corrections (konacna_oznaka for DA pairs)
        allowed = {"Algebra", "Geometrija", "Teorija brojeva", "Logika"}
        corr: dict[str, str] = {}
        where: dict[str, list[str]] = {}
        dec_rows = rev[(rev["odluka"] == "DA") & (rev["konacna"] != "")]
        for r in dec_rows.itertuples(index=False):
            if r.konacna not in allowed:
                raise SystemExit(f"Invalid label '{r.konacna}' (sheet {r.list}, row {r.excel_red})")
            for f_ in (r.family_a, r.family_b):
                where.setdefault(f_, []).append(f"sheet {r.list}, row {r.excel_red}: {r.konacna}")
        bad = {f_: w for f_, w in where.items() if len({x.split(': ')[-1] for x in w}) > 1}
        if bad:
            msg = "\n".join(f"  {f_}:\n    " + "\n    ".join(w) for f_, w in bad.items())
            raise SystemExit("The same problem received DIFFERENT final labels in several rows - make them consistent:\n" + msg)
        for r in dec_rows.itertuples(index=False):
            corr[r.family_a] = r.konacna
            corr[r.family_b] = r.konacna
        fam["label"] = [corr.get(f_, l_) for f_, l_ in zip(fam.task_family_id, fam.label)]
        changed = fam[fam.label != fam.label_original]
        sy = rev[rev["list"] == "Ista_godina"]
        unresolved = rev[(rev["odluka"] == "DA") & (rev.label_a != rev.label_b) & (rev["konacna"] == "")]
        review_info = {
            "cross_year_DA": int((cy["odluka"] == "DA").sum()), "cross_year_NE": int((cy["odluka"] == "NE").sum()),
            "cross_year_empty_treated_as_DA": n_empty,
            "same_year_reviewed": int((sy["odluka"] != "").sum()), "same_year_DA": int((sy["odluka"] == "DA").sum()),
            "same_year_NE": int((sy["odluka"] == "NE").sum()),
            "n_labels_changed": int(len(changed)),
            "label_changes": [{"family": f_, "from": a, "to": b} for f_, a, b in
                              zip(changed.task_family_id, changed.label_original, changed.label)],
            "DA_pairs_with_different_labels_unresolved": int(len(unresolved)),
        }
        if len(unresolved):
            print(f"WARNING: {len(unresolved)} DA pairs have different labels without konacna_oznaka.")
        source = f"manual review from {args.reviewed}"
    else:
        same = pairs
        source = "ALL automatic candidates (conservative) - run with --reviewed after the manual review"
        write_review_excel(pairs, fam, review_xlsx)

    # Groups: basic unit = year. Problems that are the SAME as a problem from another year (together with
    # all their same-year copies) are moved to the group of the EARLIEST year in which the problem appears.
    # Whole years are therefore never merged, and no repeated problem straddles the split.
    fid_idx = {f_: i for i, f_ in enumerate(fam.task_family_id)}
    ea = [fid_idx[a] for a in same.family_a]
    eb = [fid_idx[b] for b in same.family_b]
    G = coo_matrix((np.ones(len(ea)), (ea, eb)), shape=(len(fam), len(fam)))
    _, comp = connected_components(G, directed=False)
    fam["_comp"] = comp
    first_year = fam.groupby("_comp").year.min()
    fam["group_year"] = fam["_comp"].map(first_year)
    moved = fam[fam.group_year != fam.year]
    fam["group_id_v2"] = "year_" + fam["group_year"].astype(str)
    groups = sorted(fam.group_id_v2.unique())
    fam["fold"] = fam.group_id_v2.map({gname: i for i, gname in enumerate(groups)})
    if args.reviewed:
        # all problems linked with DA (including chains of 3+) must share one label
        da = rev[rev["odluka"] == "DA"]
        ia = [fid_idx[a] for a in da.family_a]
        ib = [fid_idx[b] for b in da.family_b]
        _, comp_da = connected_components(coo_matrix((np.ones(len(ia)), (ia, ib)), shape=(len(fam), len(fam))), directed=False)
        fam["_da"] = comp_da
        multi = fam.groupby("_da").filter(lambda g: len(g) > 1 and g.label.nunique() > 1)
        incons = [sorted(zip(g.task_family_id, g.label)) for _, g in multi.groupby("_da")]
        review_info["DA_chains_with_different_labels_after_corrections"] = len(incons)
        if incons:
            print(f"WARNING: {len(incons)} groups of the same problem (DA) still have different labels after corrections:")
            for c in incons:
                print("   ", "; ".join(f"{f_} = {l_}" for f_, l_ in c))
    review_info["n_families_moved_to_earlier_year_group"] = int(len(moved))
    review_info["moved_families"] = [{"family": f_, "year": int(y), "moved_to": int(g)}
                                     for f_, y, g in zip(moved.task_family_id, moved.year, moved.group_year)]
    fam[["task_family_id", "label", "label_original", "year", "round", "category", "group_id_v2", "fold"]].to_csv(
        args.out_dir / "image_folds_v2.csv", index=False)
    report = {
        "threshold": args.threshold, "group_source": source, "manual_review": review_info,
        "n_candidate_pairs": int(len(pairs)),
        "n_candidate_pairs_same_year_cross_category": int((pairs.same_year & (pairs.category_a != pairs.category_b)).sum()),
        "n_candidate_pairs_cross_year": int((~pairs.same_year).sum()),
        "n_candidate_pairs_label_disagreement": int((pairs.label_a != pairs.label_b).sum()),
        "n_folds": len(groups),
        "fold_sizes": fam.groupby("fold").size().astype(int).to_dict(),
        "fold_label_distribution": {int(k): v for k, v in fam.groupby("fold").label.value_counts().unstack(fill_value=0).astype(int).to_dict("index").items()},
    }
    (args.out_dir / "image_folds_v2_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    if not args.reviewed:
        print(f"\nNEXT: open {review_xlsx}, fill in sheet Medjugodisnji (DA/NE), save,"
              f" then run again with --reviewed {review_xlsx.relative_to(REPO_ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
