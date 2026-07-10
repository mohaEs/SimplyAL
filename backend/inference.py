#!/usr/bin/env python3
"""
Run batch inference with a model exported by SimplyAL.

Examples
--------
python inference.py \
    --project exported_project.zip \
    --csv images.csv \
    --output predictions.csv

python inference.py \
    --model model.pt \
    --metadata metadata.json \
    --csv images.csv \
    --output predictions.csv

The input CSV must contain a `file_path` column. Relative paths are resolved
relative to the CSV file.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import torch.nn as nn
from PIL import Image, UnidentifiedImageError
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


DEFAULT_MEAN = [0.485, 0.456, 0.406]
DEFAULT_STD = [0.229, 0.224, 0.225]
DEFAULT_IMAGE_SIZE = 224


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch inference for models exported by SimplyAL."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--project",
        type=Path,
        help="SimplyAL exported ZIP archive.",
    )
    source.add_argument(
        "--model",
        type=Path,
        help="Exported model.pt or model_scripted.pt.",
    )

    parser.add_argument("--metadata", type=Path, help="Optional metadata.json.")
    parser.add_argument("--config", type=Path, help="Optional inference_config.json.")
    parser.add_argument("--csv", type=Path, required=True, help="Input CSV.")
    parser.add_argument(
        "--path-column",
        default="file_path",
        help="CSV column containing image paths (default: file_path).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("predictions.csv"),
        help="Output CSV (default: predictions.csv).",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
    )
    parser.add_argument(
        "--include-probabilities",
        action="store_true",
        help="Add one probability column per class.",
    )
    return parser.parse_args()


def choose_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return torch.device(name)


def safe_torch_load(path: Path, device: torch.device) -> Any:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def normalize_model_type(model_type: str) -> str:
    value = str(model_type or "").strip().lower().replace("_", "-")
    aliases = {
        "vit": "vision-transformer",
        "vision-transformer": "vision-transformer",
        "resnet-18": "resnet18",
        "resnet-50": "resnet50",
        "efficientnet-b0": "efficientnet",
    }
    return aliases.get(value, value)


def create_model(model_type: str, num_classes: int) -> nn.Module:
    model_type = normalize_model_type(model_type)

    if model_type == "resnet18":
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    if model_type == "resnet50":
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    if model_type == "vision-transformer":
        try:
            import timm
        except ImportError as exc:
            raise RuntimeError(
                "The exported model is a Vision Transformer. Install timm: "
                "`pip install timm`."
            ) from exc
        return timm.create_model(
            "vit_base_patch16_224",
            pretrained=False,
            num_classes=num_classes,
        )

    if model_type == "dinov2":
        try:
            import timm
        except ImportError as exc:
            raise RuntimeError(
                "The exported model is DINOv2. Install timm: `pip install timm`."
            ) from exc
        return timm.create_model(
            "vit_base_patch14_dinov2.lvd142m",
            pretrained=False,
            num_classes=num_classes,
            img_size=224,
        )

    if model_type == "efficientnet":
        try:
            import timm
        except ImportError as exc:
            raise RuntimeError(
                "The exported model is EfficientNet. Install timm: "
                "`pip install timm`."
            ) from exc
        return timm.create_model(
            "efficientnet_b0",
            pretrained=False,
            num_classes=num_classes,
        )

    raise RuntimeError(
        f"Unsupported model type: {model_type!r}. "
        "For a custom architecture, export `model_scripted.pt` using the supplied "
        "export utility, or add the architecture to create_model()."
    )


def extract_project(project_zip: Path, destination: Path) -> dict[str, Path]:
    if not project_zip.exists():
        raise FileNotFoundError(f"Project archive not found: {project_zip}")

    with zipfile.ZipFile(project_zip, "r") as archive:
        archive.extractall(destination)

    files: dict[str, Path] = {}
    for name in ("model_scripted.pt", "model.pt", "metadata.json", "inference_config.json"):
        matches = list(destination.rglob(name))
        if matches:
            files[name] = matches[0]
    return files


def read_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def metadata_labels(metadata: dict[str, Any], num_classes: int) -> list[str]:
    labels = metadata.get("labels", {}).get("label_names", [])
    if isinstance(labels, list) and len(labels) == num_classes:
        return [str(label) for label in labels]
    return [f"Class {index}" for index in range(num_classes)]


def load_exported_model(
    model_path: Path,
    metadata: dict[str, Any],
    device: torch.device,
) -> tuple[nn.Module, dict[str, Any]]:
    if model_path.name == "model_scripted.pt":
        model = torch.jit.load(str(model_path), map_location=device)
        config = {
            "model_type": metadata.get("project_info", {}).get("model_type", "scripted"),
            "num_classes": metadata.get("labels", {}).get("num_classes"),
        }
        return model.eval(), config

    payload = safe_torch_load(model_path, device)

    if isinstance(payload, nn.Module):
        return payload.to(device).eval(), {}

    if not isinstance(payload, dict):
        raise RuntimeError("Unrecognized exported model format.")

    state_dict = (
        payload.get("model_state")
        or payload.get("model_state_dict")
        or payload.get("state_dict")
    )
    config = payload.get("model_config", {})

    if state_dict is None and all(isinstance(key, str) for key in payload):
        state_dict = payload

    if state_dict is None:
        raise RuntimeError("No model state dictionary was found in the model file.")

    model_type = (
        config.get("model_type")
        or metadata.get("project_info", {}).get("model_type")
    )
    num_classes = (
        config.get("num_classes")
        or metadata.get("labels", {}).get("num_classes")
        or metadata.get("project_info", {}).get("num_classes")
    )

    if not model_type or not num_classes:
        raise RuntimeError(
            "The export does not contain model_type and num_classes. "
            "Re-export the project with inference metadata."
        )

    model = create_model(str(model_type), int(num_classes))
    incompatible = model.load_state_dict(state_dict, strict=False)

    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "The model architecture does not match the exported weights.\n"
            f"Missing keys: {incompatible.missing_keys}\n"
            f"Unexpected keys: {incompatible.unexpected_keys}"
        )

    return model.to(device).eval(), {
        **config,
        "model_type": model_type,
        "num_classes": int(num_classes),
    }


class ImageCsvDataset(Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        csv_path: Path,
        path_column: str,
        transform: transforms.Compose,
    ) -> None:
        self.dataframe = dataframe.reset_index(drop=True)
        self.csv_dir = csv_path.resolve().parent
        self.path_column = path_column
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataframe)

    def resolve_path(self, raw_path: str) -> Path:
        path = Path(str(raw_path)).expanduser()
        if not path.is_absolute():
            path = self.csv_dir / path
        return path.resolve()

    def __getitem__(self, index: int) -> dict[str, Any]:
        raw_path = self.dataframe.loc[index, self.path_column]
        path = self.resolve_path(raw_path)

        try:
            with Image.open(path) as image:
                image = image.convert("RGB")
                tensor = self.transform(image)
            return {
                "index": index,
                "tensor": tensor,
                "resolved_path": str(path),
                "error": "",
            }
        except (FileNotFoundError, UnidentifiedImageError, OSError) as exc:
            return {
                "index": index,
                "tensor": torch.zeros(3, DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE),
                "resolved_path": str(path),
                "error": str(exc),
            }


def collate_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "indices": [item["index"] for item in batch],
        "tensors": torch.stack([item["tensor"] for item in batch]),
        "resolved_paths": [item["resolved_path"] for item in batch],
        "errors": [item["error"] for item in batch],
    }


def main() -> int:
    args = parse_args()
    device = choose_device(args.device)

    if not args.csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {args.csv}")

    dataframe = pd.read_csv(args.csv)
    if args.path_column not in dataframe.columns:
        raise ValueError(
            f"Column {args.path_column!r} was not found. "
            f"Available columns: {list(dataframe.columns)}"
        )

    with tempfile.TemporaryDirectory(prefix="simplyal_inference_") as temp_dir:
        temp_path = Path(temp_dir)

        if args.project:
            project_files = extract_project(args.project, temp_path)
            model_path = project_files.get("model_scripted.pt") or project_files.get("model.pt")
            metadata_path = args.metadata or project_files.get("metadata.json")
            config_path = args.config or project_files.get("inference_config.json")
            if model_path is None:
                raise RuntimeError("The project archive contains no exported model.")
        else:
            model_path = args.model
            metadata_path = args.metadata
            config_path = args.config

        metadata = read_json(metadata_path)
        inference_config = read_json(config_path)
        model, model_config = load_exported_model(model_path, metadata, device)

        image_size = int(
            inference_config.get(
                "image_size",
                model_config.get("image_size", DEFAULT_IMAGE_SIZE),
            )
        )
        mean = inference_config.get("normalize_mean", DEFAULT_MEAN)
        std = inference_config.get("normalize_std", DEFAULT_STD)

        transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

        num_classes = (
            model_config.get("num_classes")
            or metadata.get("labels", {}).get("num_classes")
        )
        if not num_classes:
            raise RuntimeError("Could not determine the number of classes.")

        labels = metadata_labels(metadata, int(num_classes))
        dataset = ImageCsvDataset(dataframe, args.csv, args.path_column, transform)
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_batch,
            pin_memory=device.type == "cuda",
        )

        results: dict[int, dict[str, Any]] = {}

        with torch.inference_mode():
            for batch in loader:
                valid_positions = [
                    position
                    for position, error in enumerate(batch["errors"])
                    if not error
                ]

                for position, index in enumerate(batch["indices"]):
                    if batch["errors"][position]:
                        results[index] = {
                            "resolved_file_path": batch["resolved_paths"][position],
                            "prediction_index": None,
                            "prediction_label": None,
                            "prediction_confidence": None,
                            "inference_error": batch["errors"][position],
                        }

                if not valid_positions:
                    continue

                tensors = batch["tensors"][valid_positions].to(
                    device, non_blocking=True
                )
                logits = model(tensors)
                if isinstance(logits, (tuple, list)):
                    logits = logits[0]
                if isinstance(logits, dict):
                    logits = logits.get("logits")
                if logits is None:
                    raise RuntimeError("The model did not return logits.")

                probabilities = torch.softmax(logits, dim=1).cpu()
                confidences, predictions = probabilities.max(dim=1)

                for output_position, source_position in enumerate(valid_positions):
                    index = batch["indices"][source_position]
                    prediction_index = int(predictions[output_position])
                    row = {
                        "resolved_file_path": batch["resolved_paths"][source_position],
                        "prediction_index": prediction_index,
                        "prediction_label": labels[prediction_index],
                        "prediction_confidence": float(confidences[output_position]),
                        "inference_error": "",
                    }
                    if args.include_probabilities:
                        for class_index, label in enumerate(labels):
                            safe_label = "".join(
                                character if character.isalnum() else "_"
                                for character in label
                            ).strip("_")
                            row[f"probability_{class_index}_{safe_label}"] = float(
                                probabilities[output_position, class_index]
                            )
                    results[index] = row

        prediction_frame = pd.DataFrame.from_dict(results, orient="index").sort_index()
        output_frame = pd.concat(
            [dataframe.reset_index(drop=True), prediction_frame.reset_index(drop=True)],
            axis=1,
        )

        args.output.parent.mkdir(parents=True, exist_ok=True)
        output_frame.to_csv(args.output, index=False)

        success_count = int((output_frame["inference_error"] == "").sum())
        failure_count = len(output_frame) - success_count
        print(f"Device: {device}")
        print(f"Predictions written to: {args.output.resolve()}")
        print(f"Successful: {success_count}; failed: {failure_count}")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
