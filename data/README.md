# Data

## Text corpus (`data/interim/text_expanded/`)

| File | Content |
|---|---|
| `problems_clean.jsonl` | cleaned corpus, 2974 problems (input of `src/data/build_text_v2_groups.py`) |
| `problems_ml_ready_v2.jsonl` | final corpus used in all text experiments: 2974 problems with the final expert labels and near-duplicate groups |

Fields of `problems_ml_ready_v2.jsonl` (the first fifteen are also in `problems_clean.jsonl`):

| Field | Meaning |
|---|---|
| `id` | problem id `<year>-<level>-<category>-<grade>-<number>`, e.g. `2002-Drzavno-A-1-1` |
| `kategorija`, `razred`, `nivo` | competition category, grade, level (`opstinsko` municipal, `okruzno` district, `drzavno` state) |
| `oblast` | final expert area: `Algebra`, `Geometrija`, `Teorija brojeva` (number theory), `Kombinatorika` |
| `tekst_zadatka`, `clean_text`, `normalized_text` | original problem text (LaTeX), cleaned text, normalised text used for exact-duplicate detection |
| `text_hash`, `duplicate_group_id` | exact-duplicate key and group |
| `resenje` | published solution (not used by any model) |
| `prompt`, `completion` | problem/solution pair in an instruction format (not used in the paper) |
| `original_index`, `source_file` | position in the original per-competition source file |
| `oblast_original` | area before the expert review of conflicting duplicates |
| `model_text` | model input: `clean_text` without the trailing author/source attribution |
| `group_id_v2` | near-duplicate group (exact duplicates + manually confirmed near-duplicates) |

Folds used by all text models: `reports/ml/text_v2/folds.csv` (5 repetitions x 5 outer folds).

## Kangaroo image corpus

* `reports/image/final_image_annotations.csv` - one row per problem family: year, round, age category, problem number, document group and the paths of the Cyrillic and Latin images. The `label` column of this file holds the label **before** the review of repeated problems.
* `reports/image/v2/image_folds_v2.csv` - **final labels** (`label`), labels before the review (`label_original`; 30 families differ), year groups (`group_id_v2`) and the 13 leave-one-year-out folds.
* `reports/image/v2/image_folds_doc.csv` - the control split by source document (Table 6).
* `data/image_interim/ocr/kengur_ocr_v1.jsonl` - Tesseract OCR of both images of every family (`ocr_cyrillic_raw`, `ocr_cyrillic_translit` = Cyrillic OCR transliterated to Latin script, `ocr_latin`, mean word confidences `conf_cyrillic`, `conf_latin`).
* Images (5644 PNG): Zenodo, DOI **[10.5281/zenodo.24514836](https://doi.org/10.5281/zenodo.24514836)**. Unzip in the repository root; the files are placed under `data/image_interim/batch_*/task_crops/` and `data/image_interim/production_task_crops/`, as referenced by the image paths above.

Area names: `Algebra`, `Geometrija` (Geometry), `Teorija brojeva` (called *Brojevi* / Numbers in the Kangaroo competition), `Logika` (Logic).

## Source and terms of use

The problems come from Serbian mathematics competitions organised by the Mathematical Society of Serbia (Društvo matematičara Srbije, DMS): the high-school competitions (municipal, district and state level) and the Kangaroo Without Borders competition. The problem texts and the problem images are publicly available publications of the competitions and are redistributed here for research purposes; please cite the competition source together with the paper.

The expert area labels, groups, folds and all results are released under CC BY 4.0.
