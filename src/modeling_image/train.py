"""Two-phase training (frozen backbone -> unfreeze + cosine decay).

The cross-validation script used in the paper (run_image_cv.py) imports only `train_one_epoch`;
the remaining functions (early stopping on a validation split, final development training) are
kept for completeness of the original pipeline."""

from __future__ import annotations

import random
import time
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from modeling_image import model as model_mod
from modeling_image.dataset import ImageSampleDataset
from modeling_image.evaluate import aggregate_family_predictions, compute_probs, family_level_metrics

RANDOM_SEED = 42

PHASE1_EPOCHS = 2
PHASE1_LR = 1e-3
PHASE2_LR = 1e-4
MAX_TOTAL_EPOCHS = 30
EARLY_STOP_PATIENCE = 5


def set_seed(seed: int = RANDOM_SEED) -> None:
    """Reasonable, but NOT bit-identical (across GPUs/environments), reproducibility settings -
    fixes the seed for random/numpy/torch (CPU+CUDA)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_batch_size(preferred: int = 32, fallback: int = 16) -> int:
    """Returns `preferred` (32) unless a CUDA probe allocation runs out of memory - then
    `fallback` (16). Does not shrink the 256x768 canvas."""
    if not torch.cuda.is_available():
        return fallback
    try:
        probe = torch.empty((preferred, 3, 256, 768), device="cuda")
        del probe
        torch.cuda.empty_cache()
        return preferred
    except RuntimeError:
        torch.cuda.empty_cache()
        return fallback


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: torch.cuda.amp.GradScaler | None,
) -> float:
    model.train()
    total_loss = 0.0
    n_batches = 0
    for images, labels, _idx in loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(images)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
        total_loss += float(loss.item())
        n_batches += 1
    return total_loss / max(n_batches, 1)


def run_training(
    model: nn.Module,
    train_dataset,
    val_dataset: ImageSampleDataset,
    device: torch.device,
    batch_size: int,
    use_amp: bool,
    checkpoint_path,
) -> dict[str, Any]:
    """Full two-phase loop with early stopping. Returns the history (one dict per epoch) and
    information about the best epoch (by validation family macro-F1)."""
    set_seed(RANDOM_SEED)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=False)

    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda") if (use_amp and device.type == "cuda") else None

    history: list[dict[str, Any]] = []
    best_val_family_f1 = -1.0
    best_epoch = -1
    epochs_without_improvement = 0
    global_epoch = 0

    # ---- PHASE 1: frozen backbone, classification head only ----
    model_mod.freeze_backbone(model)
    optimizer = torch.optim.AdamW(model_mod.trainable_parameters(model), lr=PHASE1_LR)

    for _ in range(PHASE1_EPOCHS):
        global_epoch += 1
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler)
        val_metrics, per_family = _evaluate_family(model, val_dataset, device, batch_size)
        elapsed = time.time() - t0

        row = _history_row(global_epoch, "phase1", train_loss, val_metrics, elapsed)
        history.append(row)
        print(f"  [epoch {global_epoch:02d} | phase1] loss={train_loss:.4f} val_family_macroF1={val_metrics['macro_f1']:.4f} ({elapsed:.1f}s)")

        if val_metrics["macro_f1"] > best_val_family_f1:
            best_val_family_f1 = val_metrics["macro_f1"]
            best_epoch = global_epoch
            torch.save(model.state_dict(), checkpoint_path)

    # ---- PHASE 2: unfreeze all, cosine decay ----
    model_mod.unfreeze_all(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=PHASE2_LR)
    phase2_max_epochs = max(1, MAX_TOTAL_EPOCHS - PHASE1_EPOCHS)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=phase2_max_epochs)

    for _ in range(phase2_max_epochs):
        global_epoch += 1
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler)
        val_metrics, per_family = _evaluate_family(model, val_dataset, device, batch_size)
        scheduler.step()
        elapsed = time.time() - t0

        row = _history_row(global_epoch, "phase2", train_loss, val_metrics, elapsed, lr=scheduler.get_last_lr()[0])
        history.append(row)
        print(
            f"  [epoch {global_epoch:02d} | phase2] loss={train_loss:.4f} "
            f"val_family_macroF1={val_metrics['macro_f1']:.4f} lr={row['lr']:.2e} ({elapsed:.1f}s)"
        )

        if val_metrics["macro_f1"] > best_val_family_f1 + 1e-9:
            best_val_family_f1 = val_metrics["macro_f1"]
            best_epoch = global_epoch
            epochs_without_improvement = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= EARLY_STOP_PATIENCE:
                print(f"  [early stopping] {EARLY_STOP_PATIENCE} epochs without improvement - stopping at epoch {global_epoch}.")
                break

    return {
        "history": history,
        "best_epoch": best_epoch,
        "best_val_family_macro_f1": best_val_family_f1,
        "total_epochs_run": global_epoch,
    }


def run_final_training(
    model: nn.Module,
    dev_dataset,
    device: torch.device,
    batch_size: int,
    use_amp: bool,
    checkpoint_path,
    *,
    phase1_epochs: int = PHASE1_EPOCHS,
    phase1_lr: float = PHASE1_LR,
    phase2_epochs_to_run: int = 12,
    phase2_lr: float = PHASE2_LR,
    cosine_t_max: int = 28,
) -> dict[str, Any]:
    """Final development training (training data only) - NO validation DataLoader, NO early
    stopping, NO best-checkpoint selection. Trains EXACTLY `phase1_epochs + phase2_epochs_to_run`
    epochs; the checkpoint is saved ONLY after the last epoch.

    The cosine scheduler horizon (`T_max=cosine_t_max`, 28) is deliberately SEPARATE from the
    number of phase-2 epochs actually run (`phase2_epochs_to_run=12`): `scheduler.step()` is called
    after EVERY phase-2 epoch with `eta_min=0`, so the learning rate decays only part of the way
    (about 61% of the initial value after 12 epochs). The lr recorded for epoch E is the value
    after `scheduler.step()`, i.e. the lr used in epoch E+1."""
    set_seed(RANDOM_SEED)

    train_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=False)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda") if (use_amp and device.type == "cuda") else None

    history: list[dict[str, Any]] = []
    global_epoch = 0

    # ---- PHASE 1: frozen backbone, classification head only ----
    model_mod.freeze_backbone(model)
    optimizer = torch.optim.AdamW(model_mod.trainable_parameters(model), lr=phase1_lr)

    for _ in range(phase1_epochs):
        global_epoch += 1
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler)
        elapsed = time.time() - t0
        row = _final_history_row(global_epoch, "phase1", train_loss, elapsed, lr=phase1_lr)
        history.append(row)
        print(f"  [epoch {global_epoch:02d} | phase1] loss={train_loss:.4f} ({elapsed:.1f}s)")

    # ---- PHASE 2: unfreeze all, cosine decay (T_max=cosine_t_max, only the first phase2_epochs_to_run steps) ----
    model_mod.unfreeze_all(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=phase2_lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cosine_t_max)

    for _ in range(phase2_epochs_to_run):
        global_epoch += 1
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler)
        scheduler.step()
        elapsed = time.time() - t0
        row = _final_history_row(global_epoch, "phase2", train_loss, elapsed, lr=scheduler.get_last_lr()[0])
        history.append(row)
        print(f"  [epoch {global_epoch:02d} | phase2] loss={train_loss:.4f} lr={row['lr']:.6e} ({elapsed:.1f}s)")

    # Checkpoint ONLY after the last (14th) epoch - no best-checkpoint selection.
    torch.save(model.state_dict(), checkpoint_path)

    return {
        "history": history,
        "total_epochs_run": global_epoch,
        "final_train_loss": history[-1]["train_loss"] if history else None,
    }


def _final_history_row(epoch, phase, train_loss, elapsed, lr):
    return {
        "epoch": epoch,
        "phase": phase,
        "train_loss": train_loss,
        "lr": lr,
        "elapsed_sec": round(elapsed, 2),
    }


def _evaluate_family(model, val_dataset, device, batch_size):
    probs, _labels = compute_probs(model, val_dataset, device, batch_size=batch_size)
    per_family = aggregate_family_predictions(probs, val_dataset)
    metrics = family_level_metrics(per_family)
    return metrics, per_family


def _history_row(epoch, phase, train_loss, val_metrics, elapsed, lr=None):
    return {
        "epoch": epoch,
        "phase": phase,
        "train_loss": train_loss,
        "val_family_accuracy": val_metrics["accuracy"],
        "val_family_macro_precision": val_metrics["macro_precision"],
        "val_family_macro_recall": val_metrics["macro_recall"],
        "val_family_macro_f1": val_metrics["macro_f1"],
        "val_family_weighted_f1": val_metrics["weighted_f1"],
        "lr": lr if lr is not None else PHASE1_LR,
        "elapsed_sec": round(elapsed, 2),
    }
