"""Train the three-label DenseNet on the NIH ChestX-ray14 image-level labels."""
from __future__ import annotations

import argparse
import csv
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

LABELS = ("Pneumonia", "Pneumothorax", "Effusion")
PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ChestXrayDataset(Dataset):
    def __init__(self, rows: list[tuple[Path, str, tuple[bool, ...]]], transform):
        self.rows = rows
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        image_path, _, target = self.rows[index]
        with Image.open(image_path) as image:
            image = self.transform(image.convert("RGB"))
        return image, torch.tensor(target, dtype=torch.float32)


def load_records(data_dir: Path):
    csv_path = data_dir / "Data_Entry_2017.csv"
    train_list = data_dir / "train_val_list.txt"
    test_list = data_dir / "test_list.txt"
    for path in (csv_path, train_list, test_list):
        if not path.is_file():
            raise FileNotFoundError(f"Required NIH dataset file not found: {path}")

    records = {}
    with csv_path.open(newline="", encoding="utf-8-sig") as file:
        for row in csv.DictReader(file):
            image_id = row.get("Image Index")
            if image_id:
                findings = set((row.get("Finding Labels") or "").split("|"))
                records[image_id] = (
                    row.get("Patient ID") or "",
                    tuple(label in findings for label in LABELS),
                )
    train_ids = set(train_list.read_text(encoding="utf-8-sig").splitlines())
    test_ids = set(test_list.read_text(encoding="utf-8-sig").splitlines())
    if train_ids & test_ids:
        raise ValueError("Official NIH train/validation and test lists overlap")
    if (train_ids | test_ids) - records.keys():
        raise ValueError("NIH split lists contain image IDs missing from the label CSV")

    image_paths = {
        path.name: path
        for path in data_dir.glob("images_*/images/*.png")
        if path.is_file()
    }
    missing_train = train_ids - image_paths.keys()
    missing_test = test_ids - image_paths.keys()
    train_rows = [
        (image_paths[image_id], *records[image_id])
        for image_id in sorted(train_ids - missing_train)
    ]
    test_rows = [
        (image_paths[image_id], *records[image_id])
        for image_id in sorted(test_ids - missing_test)
    ]
    if not train_rows or not test_rows:
        raise ValueError("NIH image folders must contain images from both official splits")
    train_patients = {row[1] for row in train_rows}
    test_patients = {row[1] for row in test_rows}
    if train_patients & test_patients:
        raise ValueError("An NIH patient appears in both official train/validation and test lists")
    return train_rows, test_rows, len(missing_train), len(missing_test)


def split_train_validation(rows: list[tuple[Path, str, tuple[bool, ...]]], seed: int):
    patient_ids = sorted({row[1] for row in rows})
    random.Random(seed).shuffle(patient_ids)
    validation_patients = set(patient_ids[:max(1, round(len(patient_ids) * 0.1))])
    train_rows = [row for row in rows if row[1] not in validation_patients]
    validation_rows = [row for row in rows if row[1] in validation_patients]
    if not train_rows or not validation_rows:
        raise ValueError("Could not create non-empty patient-level train and validation splits")
    return train_rows, validation_rows


def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    count = 0
    for step, (images, targets) in enumerate(loader, start=1):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(images)
                loss = criterion(logits, targets)
            if training:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        count += batch_size
        if training and step % 500 == 0:
            print(f"  batches {step}/{len(loader)} loss={total_loss / count:.4f}", flush=True)
    return total_loss / count


def evaluate_test_auc(model, loader, device):
    model.eval()
    scores = []
    targets = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(images)
            scores.append(torch.sigmoid(logits).cpu().numpy())
            targets.append(labels.numpy())

    scores = np.concatenate(scores)
    targets = np.concatenate(targets)
    aucs = {}
    for column, label in enumerate(LABELS):
        actual = targets[:, column].astype(bool)
        positives = int(actual.sum())
        negatives = len(actual) - positives
        if not positives or not negatives:
            aucs[label] = None
            continue
        order = np.argsort(scores[:, column], kind="mergesort")
        sorted_scores = scores[order, column]
        starts = np.r_[0, np.flatnonzero(np.diff(sorted_scores)) + 1]
        ends = np.r_[starts[1:], len(sorted_scores)]
        average_ranks = np.repeat((starts + 1 + ends) / 2, ends - starts)
        ranks = np.empty_like(average_ranks)
        ranks[order] = average_ranks
        aucs[label] = float(
            (ranks[actual].sum() - positives * (positives + 1) / 2)
            / (positives * negatives)
        )
    return aucs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True, help="Root of the full NIH dataset")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4, help="Use a small batch for GPUs with limited VRAM")
    parser.add_argument("--workers", type=int, default=0, help="DataLoader subprocesses; 0 avoids Windows shared-memory pressure")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "ml" / "saved_models")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.workers < 0:
        parser.error("epochs and batch-size must be positive; workers cannot be negative")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

    full_train_rows, test_rows, missing_train, missing_test = load_records(args.data_dir)
    train_rows, validation_rows = split_train_validation(full_train_rows, args.seed)
    print(
        f"Images: train={len(train_rows)}, validation={len(validation_rows)}, "
        f"held-out official test={len(test_rows)}"
    )
    if missing_train or missing_test:
        print(
            f"WARNING: skipped unavailable images: train/validation={missing_train}, "
            f"official test={missing_test}",
            flush=True,
        )
    print("Labels: " + ", ".join(
        f"{label}={sum(row[2][column] for row in train_rows)}"
        for column, label in enumerate(LABELS)
    ))

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    train_data = ChestXrayDataset(train_rows, transform)
    validation_data = ChestXrayDataset(validation_rows, transform)
    test_data = ChestXrayDataset(test_rows, transform)
    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "pin_memory": device.type == "cuda" and os.name != "nt",
        "persistent_workers": args.workers > 0,
    }
    train_loader = DataLoader(train_data, shuffle=True, **loader_options)
    validation_loader = DataLoader(validation_data, shuffle=False, **loader_options)
    test_loader = DataLoader(test_data, shuffle=False, **loader_options)

    weights = models.DenseNet121_Weights.DEFAULT
    model = models.densenet121(weights=weights)
    model.classifier = nn.Linear(model.classifier.in_features, len(LABELS))
    model.to(device)

    positives = torch.tensor(
        [sum(row[2][column] for row in train_rows) for column in range(len(LABELS))],
        dtype=torch.float32,
        device=device,
    )
    negatives = len(train_rows) - positives
    criterion = nn.BCEWithLogitsLoss(pos_weight=negatives / positives.clamp_min(1))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output_dir / "densenet121_clinadapt.pt"
    best_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        started = time.monotonic()
        train_loss = run_epoch(model, train_loader, criterion, device, optimizer, scaler)
        validation_loss = run_epoch(model, validation_loader, criterion, device)
        print(
            f"Epoch {epoch}/{args.epochs}: train_loss={train_loss:.4f} "
            f"validation_loss={validation_loss:.4f} elapsed={time.monotonic() - started:.0f}s",
            flush=True,
        )
        if validation_loss < best_loss:
            temporary = checkpoint.with_suffix(".pt.tmp")
            torch.save(model.cpu().state_dict(), temporary)
            os.replace(temporary, checkpoint)
            model.to(device)
            best_loss = validation_loss
            print(f"Saved best checkpoint: {checkpoint}", flush=True)

    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    test_auc = evaluate_test_auc(model, test_loader, device)
    print("Official held-out test AUROC: " + ", ".join(
        f"{label}={value:.4f}" if value is not None else f"{label}=undefined"
        for label, value in test_auc.items()
    ))
    print("These weak-label research weights are not clinically validated or calibrated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
