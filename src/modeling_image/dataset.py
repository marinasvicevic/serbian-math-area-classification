"""Dataset classes for the Kangaroo image corpus.

TRAINING UNIT = a single image (the Cyrillic and the Latin image of the same family enter
training as TWO separate samples with the same label).

Evaluation uses the SAME image-level structure (simple forward pass); aggregation to the
problem-family level (averaging the softmax outputs of the Cyrillic and Latin images) is done
afterwards in `evaluate.py`. `ImageSampleDataset` keeps the metadata needed for that
(`task_family_id`, `script`).

GUARD: `ImageSampleDataset` receives `expected_split` (one split name or a collection of allowed
split names) and checks for EVERY family record that `record["split"]` is allowed; a record with an
unexpected split (e.g. "test") raises an error immediately. This is a deliberate, redundant
protection against test data leaking into training.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Iterable

from PIL import Image
from torch.utils.data import Dataset

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

CLASSES = ["Algebra", "Geometrija", "Teorija brojeva", "Logika"]
LABEL_TO_IDX = {label: idx for idx, label in enumerate(CLASSES)}
IDX_TO_LABEL = {idx: label for label, idx in LABEL_TO_IDX.items()}

_SCRIPT_FIELDS = (
    ("cyrillic", "cyrillic_image_path", "cyrillic_example_id"),
    ("latin", "latin_image_path", "latin_example_id"),
)


class TestSplitLeakageError(RuntimeError):
    """Raised if a test (or unexpected) split reaches a training Dataset."""


class ImageSampleDataset(Dataset):
    """Image-level dataset: one sample = one image (Cyrillic OR Latin) of one problem family.
    `self.samples[i]` keeps the full metadata (for family-level aggregation in evaluate.py)."""

    def __init__(
        self,
        family_records: list[dict[str, Any]],
        transform: Callable[[Image.Image], Any],
        expected_split: str | Iterable[str],
        *,
        allow_test: bool = False,
    ) -> None:
        allowed_splits = {expected_split} if isinstance(expected_split, str) else set(expected_split)
        if "test" in allowed_splits and not allow_test:
            raise ValueError(
                "ImageSampleDataset does not allow 'test' in expected_split unless "
                "allow_test=True is passed explicitly (ONLY for a one-off, locked "
                "final test evaluation). The test split is quarantined and must never "
                "be allowed implicitly or by default."
            )
        bad = [r["task_family_id"] for r in family_records if r.get("split") not in allowed_splits]
        if bad:
            raise TestSplitLeakageError(
                f"ImageSampleDataset(expected_split={sorted(allowed_splits)!r}) received records with "
                f"a disallowed split: {bad[:10]}{'...' if len(bad) > 10 else ''} "
                "(GUARD: a test split must NEVER reach a training Dataset)."
            )
        unknown_labels = {r["label"] for r in family_records} - set(LABEL_TO_IDX)
        if unknown_labels:
            raise ValueError(f"Unknown classes in family_records: {unknown_labels}")

        self.transform = transform
        self.expected_split = allowed_splits
        self.samples: list[dict[str, Any]] = []
        for fam in family_records:
            for script, path_key, example_id_key in _SCRIPT_FIELDS:
                self.samples.append(
                    {
                        "task_family_id": fam["task_family_id"],
                        "document_group_id": fam["document_group_id"],
                        "script": script,
                        "path": str(REPO_ROOT / fam[path_key]),
                        "example_id": fam[example_id_key],
                        "label": fam["label"],
                        "label_idx": LABEL_TO_IDX[fam["label"]],
                        "year": fam.get("year"),
                        "round": fam.get("round"),
                        "category": fam.get("category"),
                        "task_number": fam.get("task_number"),
                    }
                )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        img = Image.open(sample["path"])
        img = self.transform(img)
        return img, sample["label_idx"], idx


def load_split_families(split_json_path: Path) -> list[dict[str, Any]]:
    import json

    payload = json.loads(split_json_path.read_text(encoding="utf-8"))
    return payload["families"]


def filter_by_split(families: list[dict[str, Any]], split_name: str) -> list[dict[str, Any]]:
    return [f for f in families if f["split"] == split_name]
