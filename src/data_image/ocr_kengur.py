"""OCR of all Kangaroo images (Tesseract 5).

For every problem family the Cyrillic image is OCR-ed with the `srp` model and the Latin image
with `srp_latn`; the Cyrillic output is transliterated to Latin script, and everything is written
to one JSONL file. The output is used by:
  - build_image_folds_v2.py (finding the same problem across categories/years),
  - image_ocr_text_cv.py   (OCR -> text classifier, same folds as the CNN).

Installation (Windows):
  1) Tesseract 5: https://github.com/UB-Mannheim/tesseract/wiki
     select the languages "Serbian" and "Serbian (Latin)" during installation,
     or download srp.traineddata and srp_latn.traineddata (tessdata_best) into the tessdata folder.
  2) pip install pytesseract
  3) if tesseract is not on PATH: --tesseract-cmd "C:/Program Files/Tesseract-OCR/tesseract.exe"

Command (from the repository root):
    python src/data_image/ocr_kengur.py
    python src/data_image/ocr_kengur.py --limit 20      # quick check
    python src/data_image/ocr_kengur.py --workers 8     # 8 families in parallel (much faster)

Output: data/image_interim/ocr/kengur_ocr_v1.jsonl  (+ reports/image/v2/ocr_summary.json)
Note: OCR is deterministic and does NOT use labels, so it can be run on the whole corpus
(including test parts) without leakage.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd
from PIL import Image, ImageOps

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
ANNOT = REPO_ROOT / "reports" / "image" / "final_image_annotations.csv"
OUT = REPO_ROOT / "data" / "image_interim" / "ocr" / "kengur_ocr_v1.jsonl"
SUMMARY = REPO_ROOT / "reports" / "image" / "v2" / "ocr_summary.json"

CYR2LAT = {
    "А": "A", "Б": "B", "В": "V", "Г": "G", "Д": "D", "Ђ": "Đ", "Е": "E", "Ж": "Ž", "З": "Z", "И": "I",
    "Ј": "J", "К": "K", "Л": "L", "Љ": "Lj", "М": "M", "Н": "N", "Њ": "Nj", "О": "O", "П": "P", "Р": "R",
    "С": "S", "Т": "T", "Ћ": "Ć", "У": "U", "Ф": "F", "Х": "H", "Ц": "C", "Ч": "Č", "Џ": "Dž", "Ш": "Š",
}
CYR2LAT.update({k.lower(): v.lower() for k, v in CYR2LAT.items()})


def transliterate(text: str) -> str:
    return "".join(CYR2LAT.get(ch, ch) for ch in text)


def preprocess(path: Path, upscale: float) -> Image.Image:
    img = Image.open(path).convert("L")
    if upscale != 1.0:
        img = img.resize((round(img.width * upscale), round(img.height * upscale)), Image.LANCZOS)
    return ImageOps.autocontrast(img)


def ocr(img: Image.Image, lang: str, config: str) -> tuple[str, float]:
    import pytesseract

    data = pytesseract.image_to_data(img, lang=lang, config=config, output_type=pytesseract.Output.DICT)
    confs = [float(c) for c, w in zip(data["conf"], data["text"]) if w.strip() and float(c) >= 0]
    text = pytesseract.image_to_string(img, lang=lang, config=config)
    return re.sub(r"[ \t]+", " ", text).strip(), (sum(confs) / len(confs) if confs else 0.0)


def _ocr_family(task):
    """One problem family (both images). Runs in a separate process when --workers > 1."""
    fid, cyr_path, lat_path, upscale, config, tess_cmd = task
    if tess_cmd:
        import pytesseract

        pytesseract.pytesseract.tesseract_cmd = tess_cmd
    cyr, cc = ocr(preprocess(cyr_path, upscale), "srp", config)
    lat, cl = ocr(preprocess(lat_path, upscale), "srp_latn", config)
    return {
        "task_family_id": fid,
        "ocr_cyrillic_raw": cyr, "ocr_cyrillic_translit": transliterate(cyr),
        "ocr_latin": lat, "conf_cyrillic": round(cc, 2), "conf_latin": round(cl, 2),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotations", type=Path, default=ANNOT)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--upscale", type=float, default=2.0)
    ap.add_argument("--psm", type=int, default=6)
    ap.add_argument("--tessdata-dir", type=str, default=None)
    ap.add_argument("--tesseract-cmd", type=str, default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=1, help="number of families processed in parallel")
    args = ap.parse_args()

    if args.tesseract_cmd:
        import pytesseract

        pytesseract.pytesseract.tesseract_cmd = args.tesseract_cmd
    config = f"--psm {args.psm}" + (f" --tessdata-dir {args.tessdata_dir}" if args.tessdata_dir else "")

    fams = pd.read_csv(args.annotations)
    if args.limit:
        fams = fams.head(args.limit)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tasks = [(r.task_family_id, REPO_ROOT / r.cyrillic_image_path, REPO_ROOT / r.latin_image_path,
              args.upscale, config, args.tesseract_cmd) for r in fams.itertuples(index=False)]
    confs_c, confs_l = [], []
    if args.workers > 1:
        from multiprocessing import Pool

        pool = Pool(args.workers)
        results = pool.imap(_ocr_family, tasks, chunksize=4)  # imap preserves order
    else:
        pool = None
        results = map(_ocr_family, tasks)
    with args.out.open("w", encoding="utf-8", newline="\n") as f:
        for i, rec in enumerate(results, 1):
            confs_c.append(rec["conf_cyrillic"])
            confs_l.append(rec["conf_latin"])
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if i % 100 == 0:
                print(f"{i}/{len(tasks)}", flush=True)
    if pool is not None:
        pool.close()
        pool.join()
    summary = {"n_families": len(fams), "upscale": args.upscale, "tesseract_config": config, "workers": args.workers,
               "mean_conf_cyrillic": round(sum(confs_c) / len(confs_c), 2),
               "mean_conf_latin": round(sum(confs_l) / len(confs_l), 2)}
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
