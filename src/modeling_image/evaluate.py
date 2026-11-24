"""Family-level (primary) and image-level (diagnostic) evaluation.

The PRIMARY evaluation unit is the problem family, NOT the image:
    1. the model produces logits for the Cyrillic image;
    2. softmax -> probability vector;
    3. the same for the Latin image;
    4. the two probability vectors are averaged;
    5. argmax of the average = family prediction.

Image-level metrics (diagnostics) are NEVER used to select a configuration - they are only reported.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader

from modeling_image.dataset import CLASSES, IDX_TO_LABEL, ImageSampleDataset


@torch.no_grad()
def compute_probs(
    model: torch.nn.Module, dataset: ImageSampleDataset, device: torch.device, batch_size: int = 64
) -> tuple[np.ndarray, np.ndarray]:
    """Forward pass over the whole (image-level) dataset. Returns (probs, label_idx)
    aligned with `dataset.samples` (same order, `shuffle=False`)."""
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    all_probs = np.zeros((len(dataset), len(CLASSES)), dtype=np.float32)
    all_labels = np.zeros(len(dataset), dtype=np.int64)
    cursor = 0
    for images, labels, _idx in loader:
        images = images.to(device)
        logits = model(images)
        probs = F.softmax(logits, dim=1).cpu().numpy()
        n = probs.shape[0]
        all_probs[cursor : cursor + n] = probs
        all_labels[cursor : cursor + n] = labels.numpy()
        cursor += n
    return all_probs, all_labels


def aggregate_family_predictions(probs: np.ndarray, dataset: ImageSampleDataset) -> dict[str, dict[str, Any]]:
    """Groups image-level probabilities by `task_family_id`, averages the Cyrillic and Latin
    probability vectors; argmax of the average = family prediction."""
    per_family: dict[str, dict[str, Any]] = {}
    for i, sample in enumerate(dataset.samples):
        fid = sample["task_family_id"]
        entry = per_family.setdefault(
            fid,
            {
                "document_group_id": sample["document_group_id"],
                "true_label": sample["label"],
                "true_label_idx": sample["label_idx"],
                "year": sample.get("year"),
                "round": sample.get("round"),
                "category": sample.get("category"),
                "task_number": sample.get("task_number"),
            },
        )
        entry[f"{sample['script']}_probs"] = probs[i]

    for fid, entry in per_family.items():
        if "cyrillic_probs" not in entry or "latin_probs" not in entry:
            raise RuntimeError(f"Family {fid} does not have both (Cyrillic + Latin) images in the aggregation - unexpected.")
        avg_probs = (entry["cyrillic_probs"] + entry["latin_probs"]) / 2.0
        entry["avg_probs"] = avg_probs
        entry["pred_label_idx"] = int(np.argmax(avg_probs))
        entry["pred_label"] = IDX_TO_LABEL[entry["pred_label_idx"]]
        entry["cyrillic_pred_label"] = IDX_TO_LABEL[int(np.argmax(entry["cyrillic_probs"]))]
        entry["latin_pred_label"] = IDX_TO_LABEL[int(np.argmax(entry["latin_probs"]))]
        entry["scripts_agree"] = entry["cyrillic_pred_label"] == entry["latin_pred_label"]
    return per_family


def _classification_report_block(y_true_idx: list[int], y_pred_idx: list[int]) -> dict[str, Any]:
    labels_range = list(range(len(CLASSES)))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true_idx, y_pred_idx, labels=labels_range, average=None, zero_division=0
    )
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true_idx, y_pred_idx, labels=labels_range, average="macro", zero_division=0
    )
    weighted_f1 = f1_score(y_true_idx, y_pred_idx, labels=labels_range, average="weighted", zero_division=0)
    accuracy = float(np.mean(np.array(y_true_idx) == np.array(y_pred_idx))) if y_true_idx else 0.0
    cm = confusion_matrix(y_true_idx, y_pred_idx, labels=labels_range)

    return {
        "accuracy": accuracy,
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "per_class": {
            CLASSES[i]: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i in labels_range
        },
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_labels": CLASSES,
    }


def family_level_metrics(per_family: dict[str, dict[str, Any]]) -> dict[str, Any]:
    y_true = [entry["true_label_idx"] for entry in per_family.values()]
    y_pred = [entry["pred_label_idx"] for entry in per_family.values()]
    metrics = _classification_report_block(y_true, y_pred)
    metrics["n_families"] = len(per_family)
    return metrics


def image_level_metrics(probs: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    """Diagnostics - NEVER used to select a configuration."""
    y_pred = np.argmax(probs, axis=1).tolist()
    metrics = _classification_report_block(labels.tolist(), y_pred)
    metrics["n_images"] = len(labels)
    return metrics


def script_only_metrics(per_family: dict[str, dict[str, Any]], script: str) -> dict[str, Any]:
    """Cyrillic-only or Latin-only macro-F1 - uses the ARGMAX of a single script
    (no averaging) against the true label."""
    y_true = [entry["true_label_idx"] for entry in per_family.values()]
    y_pred = [int(np.argmax(entry[f"{script}_probs"])) for entry in per_family.values()]
    return _classification_report_block(y_true, y_pred)


def scripts_agreement_rate(per_family: dict[str, dict[str, Any]]) -> float:
    if not per_family:
        return 0.0
    n_agree = sum(1 for entry in per_family.values() if entry["scripts_agree"])
    return n_agree / len(per_family)


def build_prediction_rows(per_family: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for fid, entry in sorted(per_family.items()):
        row = {
            "task_family_id": fid,
            "true_label": entry["true_label"],
            "pred_label": entry["pred_label"],
            "cyrillic_pred_label": entry["cyrillic_pred_label"],
            "latin_pred_label": entry["latin_pred_label"],
            "scripts_agree": entry["scripts_agree"],
        }
        for i, cls in enumerate(CLASSES):
            row[f"cyrillic_prob_{cls}"] = float(entry["cyrillic_probs"][i])
        for i, cls in enumerate(CLASSES):
            row[f"latin_prob_{cls}"] = float(entry["latin_probs"][i])
        for i, cls in enumerate(CLASSES):
            row[f"avg_prob_{cls}"] = float(entry["avg_probs"][i])
        rows.append(row)
    return rows
