"""
Utilities for adding portable inference artifacts to a SimplyAL project export.
"""

from __future__ import annotations

import io
import json
from typing import Sequence
from zipfile import ZipFile

import torch
import torch.nn as nn


def add_inference_artifacts(
    zip_file: ZipFile,
    model: nn.Module,
    model_type: str,
    label_names: Sequence[str],
    *,
    image_size: int = 224,
    normalize_mean: Sequence[float] = (0.485, 0.456, 0.406),
    normalize_std: Sequence[float] = (0.229, 0.224, 0.225),
) -> dict:
    """
    Add inference_config.json and, when possible, model_scripted.pt to an
    already-open SimplyAL project ZIP.

    The TorchScript model is especially useful for custom architectures because
    it does not require inference.py to reconstruct the Python model class.
    """
    config = {
        "format_version": 1,
        "model_type": model_type,
        "num_classes": len(label_names),
        "label_names": list(label_names),
        "image_size": int(image_size),
        "color_mode": "RGB",
        "resize": [int(image_size), int(image_size)],
        "normalize_mean": list(normalize_mean),
        "normalize_std": list(normalize_std),
        "output": "logits",
        "probability_function": "softmax",
    }
    zip_file.writestr(
        "inference_config.json",
        json.dumps(config, indent=2).encode("utf-8"),
    )

    status = {
        "inference_config_added": True,
        "torchscript_added": False,
        "torchscript_error": None,
    }

    original_device = next(model.parameters()).device
    was_training = model.training

    try:
        export_model = model.to("cpu").eval()
        example = torch.zeros(1, 3, image_size, image_size)

        try:
            scripted = torch.jit.script(export_model)
        except Exception:
            scripted = torch.jit.trace(export_model, example, strict=False)

        scripted_buffer = io.BytesIO()
        torch.jit.save(scripted, scripted_buffer)
        scripted_buffer.seek(0)
        zip_file.writestr("model_scripted.pt", scripted_buffer.getvalue())
        status["torchscript_added"] = True
    except Exception as exc:
        # Standard supported architectures can still be reconstructed from
        # model.pt. Custom architectures should ideally produce TorchScript.
        status["torchscript_error"] = str(exc)
    finally:
        model.to(original_device)
        model.train(was_training)

    zip_file.writestr(
        "inference_export_status.json",
        json.dumps(status, indent=2).encode("utf-8"),
    )
    return status
