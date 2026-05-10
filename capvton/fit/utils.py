from __future__ import annotations

import json
import os
from typing import Any, List, Optional

import numpy as np
import torch
from PIL import Image


def pil_mask_to_tensor(image: Image.Image) -> torch.Tensor:
    arr = np.array(image.convert("L"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)


def pil_rgb_to_tensor(image: Image.Image) -> torch.Tensor:
    arr = np.array(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)


def tensor_to_pil_mask(tensor: torch.Tensor) -> Image.Image:
    t = tensor.detach().cpu().float()
    if t.dim() == 4:
        t = t[0]
    if t.dim() == 3:
        t = t[0]
    arr = (t.numpy() * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="L")


def tensor_to_pil_rgb(tensor: torch.Tensor) -> Image.Image:
    t = tensor.detach().cpu().float()
    if t.dim() == 4:
        t = t[0]
    arr = (t.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def tensor_to_list(tensor: torch.Tensor) -> List[float]:
    return tensor.detach().cpu().float().flatten().tolist()


def combine_masks(
    a: torch.Tensor,
    b: torch.Tensor,
    mode: str = "union",
    alpha: float = 0.5,
    threshold: float = 0.5,
) -> torch.Tensor:
    if mode == "intersection":
        return a * b
    # default: weighted union
    return torch.clamp(alpha * a + (1.0 - alpha) * b, 0.0, 1.0)


def extract_garment_foreground_mask(image: Image.Image) -> Image.Image:
    arr = np.array(image.convert("RGB"))
    bg = np.all(arr > 230, axis=2)
    mask_arr = (~bg).astype(np.uint8) * 255
    return Image.fromarray(mask_arr, mode="L")


def save_fit_artifacts(output_dir: str, fit_report=None, **tensors_or_images: Any) -> None:
    os.makedirs(output_dir, exist_ok=True)
    if fit_report is not None:
        try:
            with open(os.path.join(output_dir, "fit_report.json"), "w", encoding="utf-8") as f:
                json.dump(fit_report.to_dict(), f, indent=2, ensure_ascii=False)
        except Exception:
            pass
    for name, val in tensors_or_images.items():
        if val is None:
            continue
        try:
            if isinstance(val, torch.Tensor):
                t = val.detach().cpu().float()
                if t.dim() == 4:
                    t = t[0]
                if t.shape[0] == 3:
                    img = tensor_to_pil_rgb(t.unsqueeze(0))
                else:
                    img = tensor_to_pil_mask(t.unsqueeze(0))
                img.save(os.path.join(output_dir, f"{name}.png"))
            elif isinstance(val, Image.Image):
                val.save(os.path.join(output_dir, f"{name}.png"))
        except Exception:
            pass
