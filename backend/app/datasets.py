"""Dataset integration: NIH ChestX-ray14 sample images + CheXpert label tables.

The data directory is discovered from CLINADAPT_DATA_DIR, then common local
locations. CheXpert CSVs ship labels only; their images are not part of this
download, so those datasets are browsable/verifiable but not analysable.
"""
from __future__ import annotations

import csv
import io
import os
import random
from collections import Counter
from functools import lru_cache
from pathlib import Path
from urllib.parse import quote

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
NIH_DATASET_ID = "nih-sample"
NIH_DATASET_NAME = "NIH ChestX-ray14 sample"
MODEL_LABELS = ["Pneumonia", "Pneumothorax", "Effusion"]
CHEXPERT_COLUMNS = [
    "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", "Lung Lesion",
    "Edema", "Consolidation", "Pneumonia", "Atelectasis", "Pneumothorax",
    "Pleural Effusion", "Pleural Other", "Fracture", "Support Devices", "No Finding",
]
CHEXPERT_TO_MODEL = {"Pneumonia": "Pneumonia", "Pneumothorax": "Pneumothorax", "Pleural Effusion": "Effusion"}
CHEXPERT_FILES = {
    "chexpert-chexbert": ("CheXpert train (CheXbert labels)", "train_cheXbert.csv"),
    "chexpert-visualchexbert": ("CheXpert train (VisualCheXbert labels)", "train_visualCheXbert.csv"),
}
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg")


class DatasetNotAvailable(LookupError):
    pass


def _first_file(candidates: list[Path]) -> Path | None:
    return next((p for p in candidates if p.is_file()), None)


def _first_dir(candidates: list[Path]) -> Path | None:
    return next((p for p in candidates if p.is_dir()), None)


@lru_cache(maxsize=1)
def data_paths() -> dict:
    candidates: list[Path] = []
    env = os.getenv("CLINADAPT_DATA_DIR")
    if env:
        candidates.append(Path(env))
    candidates += [
        PROJECT_ROOT / "data",
        PROJECT_ROOT.parent / "data",
        PROJECT_ROOT.parent / "archive (1)",
        PROJECT_ROOT.parent.parent / "archive (1)",
    ]
    for root in candidates:
        if not root.is_dir():
            continue
        nih_labels = _first_file([
            root / "sample_labels.csv", root / "sample" / "sample_labels.csv",
            root / "sample" / "sample" / "sample_labels.csv",
        ])
        nih_images = _first_dir([
            root / "sample" / "images", root / "images", root / "sample" / "sample" / "images",
        ])
        chexbert = _first_file([root / "train_cheXbert.csv", root / "sample" / "train_cheXbert.csv"])
        visual = _first_file([root / "train_visualCheXbert.csv", root / "sample" / "train_visualCheXbert.csv"])
        if nih_labels or nih_images or chexbert or visual:
            return {"root": root, "nih_labels": nih_labels, "nih_images": nih_images,
                    "chexbert": chexbert, "visualchexbert": visual}
    return {"root": None, "nih_labels": None, "nih_images": None, "chexbert": None, "visualchexbert": None}


def _chexpert_path(dataset_id: str) -> Path | None:
    paths = data_paths()
    if dataset_id == "chexpert-chexbert":
        return paths["chexbert"]
    if dataset_id == "chexpert-visualchexbert":
        return paths["visualchexbert"]
    return None


@lru_cache(maxsize=2)
def _line_count(path: Path) -> int:
    with open(path, encoding="utf-8-sig") as f:
        return max(sum(1 for _ in f) - 1, 0)


# ---------------------------------------------------------------- NIH sample

@lru_cache(maxsize=1)
def _nih() -> tuple[list[str], dict[str, dict]]:
    labels_path = data_paths()["nih_labels"]
    if not labels_path:
        raise DatasetNotAvailable("NIH sample_labels.csv not found")
    order: list[str] = []
    by_id: dict[str, dict] = {}
    with open(labels_path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            image_id = (row.get("Image Index") or "").strip()
            if not image_id:
                continue
            findings = tuple(x.strip() for x in (row.get("Finding Labels") or "").split("|") if x.strip())
            by_id[image_id] = {
                "id": image_id,
                "findings": findings,
                "view": (row.get("View Position") or "").strip(),
                "age": (row.get("Patient Age") or "").strip(),
                "gender": (row.get("Patient Gender") or "").strip(),
            }
            if image_id not in order:
                order.append(image_id)
    return order, by_id


@lru_cache(maxsize=1)
def _nih_image_files() -> dict[str, Path]:
    images_dir = data_paths()["nih_images"]
    if not images_dir:
        return {}
    files: dict[str, Path] = {}
    with os.scandir(images_dir) as entries:
        for entry in entries:
            if entry.is_file() and entry.name.lower().endswith(IMAGE_SUFFIXES):
                files[entry.name] = Path(entry.path)
    return files


@lru_cache(maxsize=1)
def _nih_stats() -> dict:
    order, by_id = _nih()
    findings = Counter(f for item in by_id.values() for f in item["findings"])
    total = len(by_id)
    mapped = {}
    for label in MODEL_LABELS:
        positive = sum(1 for item in by_id.values() if label in item["findings"])
        mapped[label] = {"positive": positive, "negative": total - positive, "uncertain": 0, "blank": 0}
    return {
        "rows": len(order),
        "unique_images": total,
        "images_on_disk": len(_nih_image_files()),
        "mapped": mapped,
        "top_findings": [{"label": label, "count": count} for label, count in findings.most_common()],
    }


def _nih_item(image_id: str) -> dict:
    item = _nih()[1][image_id]
    base = f"/api/datasets/{NIH_DATASET_ID}"
    quoted = quote(image_id)
    return {
        **item,
        "ground_truth": [{"label": label, "value": 1 if label in item["findings"] else 0} for label in MODEL_LABELS],
        "image_url": f"{base}/image/{quoted}",
        "thumbnail_url": f"{base}/thumbnail/{quoted}",
    }


def nih_findings(image_id: str) -> list[str]:
    try:
        return list(_nih()[1][image_id]["findings"])
    except KeyError:
        raise DatasetNotAvailable(f"unknown NIH image {image_id}") from None


def nih_ground_truth(image_id: str) -> list[dict]:
    findings = set(nih_findings(image_id))
    return [{"label": label, "value": 1 if label in findings else 0} for label in MODEL_LABELS]


# ------------------------------------------------------------ CheXpert labels

def _label_value(raw: str | None):
    value = (raw or "").strip()
    if value in ("", "nan", "NaN"):
        return None
    try:
        number = float(value)
    except ValueError:
        return "invalid"
    if number < 0:
        return -1
    return 1 if number >= 0.5 else 0


@lru_cache(maxsize=2)
def _chexpert_stats(dataset_id: str) -> dict:
    path = _chexpert_path(dataset_id)
    if not path:
        raise DatasetNotAvailable(f"{dataset_id} label table not found")
    counts: dict[str, Counter] = {column: Counter() for column in CHEXPERT_COLUMNS}
    rows = 0
    missing: list[str] = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        missing = [column for column in CHEXPERT_COLUMNS if column not in (reader.fieldnames or [])]
        if missing:
            return {"rows": _line_count(path), "mapped": {}, "missing_columns": missing, "label_domains": {}}
        for row in reader:
            rows += 1
            for column in CHEXPERT_COLUMNS:
                counts[column][_label_value(row.get(column))] += 1
    mapped = {}
    for column, label in CHEXPERT_TO_MODEL.items():
        counter = counts[column]
        mapped[label] = {
            "positive": counter[1], "negative": counter[0],
            "uncertain": counter[-1], "blank": counter[None],
            "invalid": counter["invalid"],
        }
    domains = {
        column: {("blank" if key is None else str(key)): count for key, count in counts[column].items()}
        for column in CHEXPERT_COLUMNS
    }
    return {"rows": rows, "mapped": mapped, "missing_columns": [], "label_domains": domains}


# ------------------------------------------------------------------- public

def dataset_name(dataset_id: str) -> str:
    if dataset_id == NIH_DATASET_ID:
        return NIH_DATASET_NAME
    if dataset_id in CHEXPERT_FILES:
        return CHEXPERT_FILES[dataset_id][0]
    raise DatasetNotAvailable(f"unknown dataset {dataset_id}")


def list_datasets() -> dict:
    paths = data_paths()
    root = str(paths["root"]) if paths["root"] else None
    datasets: list[dict] = []
    if paths["nih_labels"]:
        stats = _nih_stats()
        datasets.append({
            "id": NIH_DATASET_ID,
            "name": NIH_DATASET_NAME,
            "kind": "image",
            "description": "Public NIH ChestX-ray14 sample with text-mined finding labels and images on disk.",
            "rows": stats["rows"],
            "images": stats["images_on_disk"],
            "available": bool(paths["nih_images"]),
            "mapped": stats["mapped"],
            "top_findings": stats["top_findings"][:8],
        })
    for dataset_id, (name, _) in CHEXPERT_FILES.items():
        path = _chexpert_path(dataset_id)
        if not path:
            continue
        datasets.append({
            "id": dataset_id,
            "name": name,
            "kind": "labels",
            "description": "CheXpert train label table. Annotated JPGs are not included in this download.",
            "rows": _line_count(path),
            "images": 0,
            "available": True,
            "mapped": None,
            "top_findings": [],
        })
    return {"root": root, "datasets": datasets}


def stats(dataset_id: str) -> dict:
    if dataset_id == NIH_DATASET_ID:
        table = _nih_stats()
        return {"id": dataset_id, **table, "images": table["images_on_disk"]}
    if dataset_id in CHEXPERT_FILES:
        table = _chexpert_stats(dataset_id)
        return {"id": dataset_id, "rows": table["rows"], "images": 0,
                "mapped": table["mapped"], "label_domains": table["label_domains"],
                "missing_columns": table["missing_columns"]}
    raise DatasetNotAvailable(f"unknown dataset {dataset_id}")


def items(dataset_id: str, offset: int = 0, limit: int = 12, finding: str = "", value: str = "") -> dict:
    if dataset_id != NIH_DATASET_ID:
        raise DatasetNotAvailable(f"{dataset_id} has no images in this download")
    order, by_id = _nih()
    supported = set(MODEL_LABELS)
    if finding and finding not in supported:
        raise ValueError(f"finding must be one of {sorted(supported)}")
    want_positive = value != "negative"
    ids = [
        image_id for image_id in order
        if not finding or ((finding in by_id[image_id]["findings"]) == want_positive)
    ]
    page = ids[offset:offset + limit]
    return {"dataset": dataset_id, "total": len(ids), "offset": offset, "limit": limit,
            "items": [_nih_item(image_id) for image_id in page]}


def preview(dataset_id: str, offset: int = 0, limit: int = 12) -> dict:
    if dataset_id not in CHEXPERT_FILES:
        raise DatasetNotAvailable(f"{dataset_id} is not a label table")
    path = _chexpert_path(dataset_id)
    if not path:
        raise DatasetNotAvailable(f"{dataset_id} label table not found")
    rows: list[dict] = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for index, row in enumerate(reader):
            if index < offset:
                continue
            if len(rows) >= limit:
                break
            ground_truth = []
            for column, label in CHEXPERT_TO_MODEL.items():
                parsed = _label_value(row.get(column))
                ground_truth.append({"label": label, "value": parsed if parsed in (-1, 0, 1) else None})
            rows.append({
                "id": (row.get("Path") or "").strip(),
                "path": (row.get("Path") or "").strip(),
                "sex": (row.get("Sex") or "").strip(),
                "age": (row.get("Age") or "").strip(),
                "view": (row.get("AP/PA") or "").strip(),
                "ground_truth": ground_truth,
            })
    return {"dataset": dataset_id, "total": _line_count(path), "offset": offset, "limit": limit, "items": rows}


def image_path(dataset_id: str, image_id: str) -> Path:
    if dataset_id != NIH_DATASET_ID:
        raise DatasetNotAvailable(f"{dataset_id} has no images in this download")
    path = _nih_image_files().get(image_id)
    if not path or image_id not in _nih()[1]:
        raise DatasetNotAvailable(f"unknown NIH image {image_id}")
    return path


@lru_cache(maxsize=512)
def thumbnail_bytes(image_id: str, size: int = 240) -> bytes:
    path = image_path(NIH_DATASET_ID, image_id)
    with Image.open(path) as img:
        img = img.convert("RGB")
        img.thumbnail((size, size))
        buffer = io.BytesIO()
        img.save(buffer, "JPEG", quality=82)
    return buffer.getvalue()


# --------------------------------------------------------------- verification

@lru_cache(maxsize=1)
def verify() -> dict:
    paths = data_paths()
    errors: list[str] = []
    warnings: list[str] = []
    notes: list[str] = []
    report: dict = {"data_root": str(paths["root"]) if paths["root"] else None, "datasets": {}}
    if not paths["root"]:
        return {"ok": False, "errors": ["No dataset directory found. Set CLINADAPT_DATA_DIR."],
                "warnings": [], "notes": [], "datasets": {}}

    if paths["nih_labels"] and paths["nih_images"]:
        order, by_id = _nih()
        files = _nih_image_files()
        missing = sorted(set(by_id) - set(files))
        extra = sorted(set(files) - set(by_id))
        labelled_files = sorted(set(files) & set(by_id))
        sample_ids = random.sample(labelled_files, min(25, len(labelled_files)))
        unreadable: list[str] = []
        for image_id in sample_ids:
            try:
                with Image.open(files[image_id]) as img:
                    img.verify()
            except Exception:
                unreadable.append(image_id)
        if missing:
            errors.append(f"NIH: {len(missing)} label rows have no image file")
        if extra:
            warnings.append(f"NIH: {len(extra)} image files have no label row")
        if unreadable:
            errors.append(f"NIH: {len(unreadable)} sampled images failed to open")
        duplicate = paths["nih_images"].parent / "sample" / "images"
        if duplicate.is_dir() and duplicate != paths["nih_images"]:
            notes.append(f"Duplicate image copy present at {duplicate} (ignored)")
        nih_stats = _nih_stats()
        report["datasets"][NIH_DATASET_ID] = {
            "labels": str(paths["nih_labels"]),
            "images": str(paths["nih_images"]),
            "rows": nih_stats["rows"],
            "unique_images": nih_stats["unique_images"],
            "images_on_disk": nih_stats["images_on_disk"],
            "missing_images": missing[:20],
            "unlabeled_images": extra[:20],
            "sample_images_checked": len(sample_ids),
            "unreadable_images": unreadable,
            "mapped": nih_stats["mapped"],
        }
    else:
        errors.append("NIH sample_labels.csv or sample/images not found")

    for dataset_id, (name, _) in CHEXPERT_FILES.items():
        path = _chexpert_path(dataset_id)
        if not path:
            errors.append(f"{name}: CSV not found")
            continue
        table = _chexpert_stats(dataset_id)
        if table["missing_columns"]:
            errors.append(f"{name}: missing columns {table['missing_columns']}")
        invalid = {label: row.get("invalid", 0) for label, row in (table.get("mapped") or {}).items()}
        if any(invalid.values()):
            warnings.append(f"{name}: unexpected label values {invalid}")
        report["datasets"][dataset_id] = {
            "labels": str(path),
            "rows": table["rows"],
            "mapped": table.get("mapped", {}),
            "label_domains": table.get("label_domains", {}),
            "images_available": False,
        }

    if paths["chexbert"] and paths["visualchexbert"]:
        path_mismatches = label_mismatches = 0
        with open(paths["chexbert"], newline="", encoding="utf-8-sig") as fa, \
                open(paths["visualchexbert"], newline="", encoding="utf-8-sig") as fb:
            for row_a, row_b in zip(csv.DictReader(fa), csv.DictReader(fb)):
                if (row_a.get("Path") or "").strip() != (row_b.get("Path") or "").strip():
                    path_mismatches += 1
                label_mismatches += sum(
                    1 for column in CHEXPERT_COLUMNS
                    if (row_a.get(column) or "").strip() != (row_b.get(column) or "").strip()
                )
        report["cross_check"] = {"path_mismatches": path_mismatches, "label_value_mismatches": label_mismatches}
        if path_mismatches:
            warnings.append(f"CheXpert CSVs are not row-aligned ({path_mismatches} path mismatches)")

    report.update({"ok": not errors, "errors": errors, "warnings": warnings, "notes": notes})
    return report
