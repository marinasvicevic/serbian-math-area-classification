"""CNN on the Kangaroo images in the same CV protocol as the OCR route (leave-one-year-out, 13 folds).

Reuses ImageSampleDataset, train_one_epoch, compute_probs and aggregate_family_predictions from
the `modeling_image` package.

The training recipe is FIXED in advance (no selection on test folds): ImageNet weights;
phase 1: 2 epochs with a frozen backbone, AdamW lr 1e-3; phase 2: 12 epochs of full fine-tuning,
AdamW lr 1e-4 with CosineAnnealingLR(T_max=28) stepped once per epoch; batch 32; AMP;
no augmentation; letterbox onto a white canvas (90% fill); no early stopping.

Commands used for the paper (from the repository root; PYTHONPATH=src; Windows PowerShell:
$env:PYTHONPATH="src"):
    # baseline ResNet-18, 256x768, three seeds
    python -m modeling_image.run_image_cv --arch resnet18 --height 256 --width 768 --seeds 0 1 2
    # higher resolution, one seed
    python -m modeling_image.run_image_cv --arch resnet18 --height 512 --width 1536 --seeds 0
    # ConvNeXt-T, one seed
    python -m modeling_image.run_image_cv --arch convnext_tiny --height 384 --width 1152 --seeds 0
    # document split (Table 6), one seed
    python -m modeling_image.run_image_cv --arch resnet18 --height 256 --width 768 --seeds 0 \\
        --folds reports/image/v2/image_folds_doc.csv --tag docsplit
    # quick check that everything runs
    python -m modeling_image.run_image_cv --folds-to-run 0 --phase1-epochs 1 --phase2-epochs 1 --seeds 0
After an interruption, rerun the same command: finished (seed, fold) pairs are skipped.

Output: reports/image/v2/cnn_cv/<arch>_<H>x<W>/  (fold_scores.csv, oof_family_probs.csv, summary.json)
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models
from torchvision import transforms as T

from modeling_image.dataset import CLASSES, REPO_ROOT, ImageSampleDataset
from modeling_image.evaluate import aggregate_family_predictions, compute_probs, family_level_metrics, script_only_metrics
from modeling_image.train import train_one_epoch
from modeling_image.transforms import IMAGENET_MEAN, IMAGENET_STD, LetterboxToCanvas

FOLDS = REPO_ROOT / "reports" / "image" / "v2" / "image_folds_v2.csv"
ANNOT = REPO_ROOT / "reports" / "image" / "final_image_annotations.csv"
OUT_ROOT = REPO_ROOT / "reports" / "image" / "v2" / "cnn_cv"


def build(arch: str, n_classes: int = 4) -> tuple[nn.Module, str]:
    """Returns the model and the parameter-name prefix of the classification head."""
    if arch == "resnet18":
        m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        m.fc = nn.Linear(m.fc.in_features, n_classes)
        return m, "fc."
    if arch == "resnet50":
        m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        m.fc = nn.Linear(m.fc.in_features, n_classes)
        return m, "fc."
    if arch == "efficientnet_v2_s":
        m = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, n_classes)
        return m, "classifier."
    if arch == "convnext_tiny":
        m = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, n_classes)
        return m, "classifier."
    raise ValueError(arch)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="resnet18", choices=["resnet18", "resnet50", "efficientnet_v2_s", "convnext_tiny"])
    ap.add_argument("--height", type=int, default=256)
    ap.add_argument("--width", type=int, default=768)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--phase1-epochs", type=int, default=2)
    ap.add_argument("--phase2-epochs", type=int, default=12)
    ap.add_argument("--cosine-tmax", type=int, default=28)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--folds-to-run", type=int, nargs="*", default=None)
    ap.add_argument("--folds", type=Path, default=FOLDS)
    ap.add_argument("--out-root", type=Path, default=OUT_ROOT, help="where results are stored")
    ap.add_argument("--workers", type=int, default=2, help="DataLoader num_workers")
    ap.add_argument("--tag", default="", help="suffix of the output folder name (e.g. docsplit)")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    folds = pd.read_csv(args.folds)
    # labels are taken from the fold file (which contains the expert corrections), not from the annotation file
    annot = pd.read_csv(ANNOT).drop(columns=["label"]).merge(
        folds[["task_family_id", "label", "group_id_v2", "fold"]], on="task_family_id")
    tf = T.Compose([LetterboxToCanvas(args.height, args.width), T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    out_dir = args.out_root / (f"{args.arch}_{args.height}x{args.width}" + (f"_{args.tag}" if args.tag else ""))
    out_dir.mkdir(parents=True, exist_ok=True)

    # Resume after an interruption: finished (seed, fold) pairs are skipped.
    fold_rows, prob_rows = [], []
    if (out_dir / "fold_scores.csv").exists():
        fold_rows = pd.read_csv(out_dir / "fold_scores.csv").to_dict("records")
        prob_rows = pd.read_csv(out_dir / "oof_family_probs.csv").to_dict("records")
    done = {(int(r["seed"]), int(r["fold"])) for r in fold_rows}
    if done:
        print(f"Resuming: {len(done)} (seed, fold) combinations already done.")
    for seed in args.seeds:
        for fold in args.folds_to_run if args.folds_to_run is not None else sorted(annot.fold.unique()):
            if (seed, int(fold)) in done:
                continue
            t0 = time.time()
            tr = annot[annot.fold != fold].assign(split="cv_train").to_dict("records")
            te = annot[annot.fold == fold].assign(split="cv_eval").to_dict("records")
            assert not {r["group_id_v2"] for r in tr} & {r["group_id_v2"] for r in te}
            set_seed(seed)
            ds_tr = ImageSampleDataset(tr, tf, "cv_train")
            ds_te = ImageSampleDataset(te, tf, "cv_eval")
            loader = DataLoader(ds_tr, batch_size=args.batch, shuffle=True, num_workers=args.workers,
                                generator=torch.Generator().manual_seed(seed))
            model, head = build(args.arch)
            model.to(device)
            crit = nn.CrossEntropyLoss()
            scaler = torch.amp.GradScaler("cuda") if use_amp else None
            for name, p in model.named_parameters():
                p.requires_grad = name.startswith(head)
            opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
            for _ in range(args.phase1_epochs):
                train_one_epoch(model, loader, opt, crit, device, scaler)
            for p in model.parameters():
                p.requires_grad = True
            opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.cosine_tmax)
            for _ in range(args.phase2_epochs):
                loss = train_one_epoch(model, loader, opt, crit, device, scaler)
                sched.step()

            probs, _ = compute_probs(model, ds_te, device, batch_size=args.batch)
            per_fam = aggregate_family_predictions(probs, ds_te)
            fam_m = family_level_metrics(per_fam)
            cyr_m, lat_m = script_only_metrics(per_fam, "cyrillic"), script_only_metrics(per_fam, "latin")
            fold_rows.append({"seed": seed, "fold": int(fold), "group": te[0]["group_id_v2"], "n_families": len(per_fam),
                              "family_macro_f1": fam_m["macro_f1"], "family_accuracy": fam_m["accuracy"],
                              "cyrillic_macro_f1": cyr_m["macro_f1"], "latin_macro_f1": lat_m["macro_f1"],
                              "final_train_loss": loss, "minutes": round((time.time() - t0) / 60, 1)})
            for fid, e in per_fam.items():
                prob_rows.append({"seed": seed, "fold": int(fold), "task_family_id": fid, "true": e["true_label"],
                                  "pred": e["pred_label"], "pred_cyr": e["cyrillic_pred_label"], "pred_lat": e["latin_pred_label"],
                                  **{f"p_{c}": float(e["avg_probs"][i]) for i, c in enumerate(CLASSES)}})
            print(f"seed {seed} fold {fold} ({te[0]['group_id_v2']}): family macroF1={fam_m['macro_f1']:.4f} "
                  f"cyr={cyr_m['macro_f1']:.4f} lat={lat_m['macro_f1']:.4f} ({fold_rows[-1]['minutes']} min)")
            pd.DataFrame(fold_rows).to_csv(out_dir / "fold_scores.csv", index=False)
            pd.DataFrame(prob_rows).to_csv(out_dir / "oof_family_probs.csv", index=False)

    fs = pd.DataFrame(fold_rows)
    pr = pd.DataFrame(prob_rows)
    if fs.empty:
        print("No results.")
        return 0
    from sklearn.metrics import f1_score
    pooled = [f1_score(g.true, g.pred, labels=CLASSES, average="macro") for _, g in pr.groupby("seed")]
    summary = {"arch": args.arch, "canvas": [args.height, args.width], "seeds": args.seeds,
               "recipe": {"phase1_epochs": args.phase1_epochs, "phase2_epochs": args.phase2_epochs, "cosine_tmax": args.cosine_tmax},
               "family_macro_f1_fold_mean": round(fs.family_macro_f1.mean(), 4),
               "family_macro_f1_fold_sd": round(fs.family_macro_f1.std(ddof=1), 4),
               "family_macro_f1_pooled_per_seed": [round(x, 4) for x in pooled],
               "cyrillic_macro_f1_fold_mean": round(fs.cyrillic_macro_f1.mean(), 4),
               "latin_macro_f1_fold_mean": round(fs.latin_macro_f1.mean(), 4)}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
