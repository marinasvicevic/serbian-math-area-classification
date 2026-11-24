# Classifying Mathematics Competition Problems by Mathematical Area Using Text and Document Images

Code, data, labels and results for the paper

> M. Svičević, N. Vučićević, A. Milenković, M. Pavković. *Classifying Mathematics Competition Problems by Mathematical Area Using Text and Document Images: Two Expert-Annotated Serbian Datasets and a Leakage-Aware Evaluation.* Submitted to *AI* (MDPI), 2026.

The repository contains two expert-annotated Serbian datasets and everything needed to reproduce every table and figure of the paper:

| | Text corpus | Kangaroo image corpus |
|---|---|---|
| Source | Serbian high-school mathematics competitions (municipal, district, state) | "Kangaroo Without Borders" (Kengur bez granica), Serbia |
| Unit | problem (LaTeX text, Latin script) | problem family = the same problem in Cyrillic and in Latin script |
| Size | 2974 problems | 2822 families, 5644 images |
| Areas | Algebra, Geometry, Number theory, Combinatorics | Algebra, Geometry, Numbers, Logic |
| Evaluation | nested 5×5 stratified group CV (near-duplicate groups) | leave-one-year-out CV (13 folds) |

Area labels were assigned independently by two experts who are official members of the competition committees; disagreements were resolved jointly.

## Repository layout

```
data/                     input data (see data/README.md)
  interim/text_expanded/    text corpus (JSONL)
  image_interim/ocr/        OCR output for all Kangaroo images (JSONL)
  image_interim/...         Kangaroo images (download from Zenodo, see below)
src/
  data/                   text corpus: near-duplicate groups
  data_image/             Kangaroo: OCR, year folds, document folds
  models/                 text models, OCR route, LLM/VLM classification
  modeling_image/         CNN cross-validation (ResNet-18, ConvNeXt-T)
  tools/                  paired bootstrap, figures, token-length check
reports/                  all results used in the paper (predictions, fold scores, summaries)
  data/text_v2/             near-duplicate pairs, manual review, label corrections
  ml/text_v2/               text corpus results
  image/                    image annotations; v2/ = folds and all image results
FIGURE/                   Figures 1-7 (PNG 600 dpi; PDF and SVG for Figures 2-7)
```

Area names are stored in Serbian in all data files: `Algebra`, `Geometrija` (Geometry), `Teorija brojeva` (Number theory; called *Numbers* in the Kangaroo corpus), `Kombinatorika` (Combinatorics), `Logika` (Logic).

## Data

* **Text corpus and OCR text** are included in `data/`.
* **Kangaroo images** (5644 PNG files) are archived on Zenodo: DOI **[10.5281/zenodo.24514836](https://doi.org/10.5281/zenodo.24514836)**. Unzip the archive in the repository root; the images are placed under `data/image_interim/...`, exactly as referenced by `reports/image/final_image_annotations.csv` (columns `cyrillic_image_path`, `latin_image_path`).
* **Final labels** for the image corpus are in `reports/image/v2/image_folds_v2.csv` (column `label`; `label_original` holds the label before the review of repeated problems).

The competition problems are publicly available publications of the competition organiser, the Mathematical Society of Serbia (Društvo matematičara Srbije); see `data/README.md`.

## Environment

Python 3.10 or newer. Install the dependencies with

```
pip install -r requirements.txt
```

`requirements.txt` pins the direct dependencies; `requirements-lock.txt` lists the complete local environment (`pip freeze`).

Classical models and OCR run on a CPU. BERTić/XLM-R were fine-tuned on an NVIDIA T4 (Google Colab); the CNN experiments and the local Qwen3.6-27B image runs (vLLM) used two NVIDIA L40S GPUs. OCR uses Tesseract 5.5.3 with the `srp` and `srp_latn` models. API keys are read only from environment variables (`ANTHROPIC_API_KEY`, `OPENROUTER_API_KEY`); no key is stored in the repository.

All commands are run from the repository root. The CNN scripts need `PYTHONPATH=src` (PowerShell: `$env:PYTHONPATH="src"`).

## Reproducing the paper

All result files are included, so the tables and figures can be checked without re-training. Each script below also documents its options in its header.

| Paper | Script | Result files |
|---|---|---|
| Near-duplicate groups (Sec. 3, 4) | `src/data/build_text_v2_groups.py` | `reports/data/text_v2/` |
| Table 2 (text CV), Table 5 (temporal, leave-one-level-out) | `src/models/text_paper_protocol.py` | `reports/ml/text_v2/summary.json`, `cv_fold_scores.csv`, `oof_predictions_rep0.csv`, `temporal.json`, `leave_level_out.json` |
| Table 2, BERTić / XLM-R | `src/models/text_transformer_cv.py` | `reports/ml/text_v2/transformer_*/` |
| Table 2, Claude / Qwen (text) | `src/models/llm_classify_text.py` | `reports/ml/text_v2/llm_*/` |
| Token truncation check | `src/tools/count_truncation.py` | `reports/ml/text_v2/truncation.json` |
| Table 4 (protocol sensitivity) | `src/models/text_protocol_sensitivity.py` | `reports/ml/text_v2/protocol_sensitivity.json` |
| OCR of the images | `src/data_image/ocr_kengur.py` | `data/image_interim/ocr/kengur_ocr_v1.jsonl`, `reports/image/v2/ocr_summary.json` |
| Year folds and repeated-problem review | `src/data_image/build_image_folds_v2.py` | `reports/image/v2/image_folds_v2.csv`, `cross_family_pairs*.{csv,xlsx}` |
| Document folds (Table 6) | `src/data_image/build_image_folds_doc.py` | `reports/image/v2/image_folds_doc.csv` |
| Table 3, OCR route | `src/models/image_ocr_text_cv.py` | `reports/image/v2/ocr_text_cv/` |
| Table 3, CNN | `src/modeling_image/run_image_cv.py` | `reports/image/v2/cnn_cv/` |
| Table 3, Claude / Qwen (images) | `src/models/llm_classify_image.py` | `reports/image/v2/vlm_*/` |
| Table 6 (grouping unit) | `image_ocr_text_cv.py` with `--folds`, `--purge-pairs`, `--random-control`; `run_image_cv.py` with `--folds ... --tag docsplit` | `reports/image/v2/ocr_text_cv_docsplit*/`, `ocr_text_cv_calendar_purged/`, `cnn_cv/resnet18_256x768_docsplit/` |
| Paired differences (Δ with 95% CI) | `src/tools/paired_delta.py` | console output |
| Figures 1-7 | `src/tools/make_figures.py` | `FIGURE/` |

Notes on the evaluation:

* Macro-F1 is computed over the four areas. An invalid LLM/VLM answer counts as a false negative for the true area and as a false positive for no area.
* 95% CIs are percentile intervals from a group bootstrap (2000 resamples, seed 42) of pooled out-of-fold predictions; paired differences use 5000 resamples. Groups: near-duplicate groups (text), competition years (images), documents (document-split protocols).
* Zero-shot LLM/VLM runs: Claude Opus 5.5 via the Message Batches API (temperature cannot be set for this model); Qwen3.6-27B with temperature 0 (text via OpenRouter, host Alibaba; images on a local vLLM server). The prompts are defined in `SYSTEM_PROMPT` of `llm_classify_text.py` and `llm_classify_image.py`. The `raw` column of each `predictions.csv` contains the full model answer.

## Licence

* Code: MIT licence (`LICENSE`).
* Expert area labels, groups, folds and results: CC BY 4.0 (`LICENSE-DATA`).
* Problem texts and images: see `data/README.md`.

## Citation

If you use the datasets or the code, please cite the paper (full reference will be added after publication).
