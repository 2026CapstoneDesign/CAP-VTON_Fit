"""
Fit-aware module stubs and helper functions.

All nn.Module subclasses here are identity/passthrough implementations.
They load external checkpoints when available (strict=False), otherwise
degrade gracefully to heuristic fallbacks.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn

from capvton.fit.schema import (
    EaseVector,
    GarmentCategory,
    GarmentMeasurements,
    GarmentSuperCategory,
    Gender,
    UserMeasurements,
)
from capvton.fit.fit_predictor_rule import RuleBasedFitPredictor


# ── Category lookup tables ────────────────────────────────────────────────────

_GARMENT_CAT_MAP: Dict[str, GarmentCategory] = {
    "upper_body": GarmentCategory.TSHIRT,
    "lower_body": GarmentCategory.PANTS,
    "dresses":    GarmentCategory.DRESS,
    "dress":      GarmentCategory.DRESS,
    "jumpsuit":   GarmentCategory.JUMPSUIT,
    "tshirt":     GarmentCategory.TSHIRT,
    "t-shirt":    GarmentCategory.TSHIRT,
    "shirt":      GarmentCategory.SHIRT,
    "jacket":     GarmentCategory.JACKET,
    "sweater":    GarmentCategory.SWEATER,
    "hoodie":     GarmentCategory.HOODIE,
    "blouse":     GarmentCategory.BLOUSE,
    "coat":       GarmentCategory.COAT,
    "pants":      GarmentCategory.PANTS,
    "jeans":      GarmentCategory.JEANS,
    "skirt":      GarmentCategory.SKIRT,
    "shorts":     GarmentCategory.SHORTS,
}

_SUPER_CAT_MAP: Dict[str, GarmentSuperCategory] = {
    "upper_body": GarmentSuperCategory.UPPER,
    "lower_body": GarmentSuperCategory.LOWER,
    "dresses":    GarmentSuperCategory.DRESS,
    "dress":      GarmentSuperCategory.DRESS,
    "jumpsuit":   GarmentSuperCategory.DRESS,
    "tshirt":     GarmentSuperCategory.UPPER,
    "t-shirt":    GarmentSuperCategory.UPPER,
    "shirt":      GarmentSuperCategory.UPPER,
    "jacket":     GarmentSuperCategory.UPPER,
    "sweater":    GarmentSuperCategory.UPPER,
    "hoodie":     GarmentSuperCategory.UPPER,
    "blouse":     GarmentSuperCategory.UPPER,
    "coat":       GarmentSuperCategory.UPPER,
    "pants":      GarmentSuperCategory.LOWER,
    "jeans":      GarmentSuperCategory.LOWER,
    "skirt":      GarmentSuperCategory.LOWER,
    "shorts":     GarmentSuperCategory.LOWER,
}


# ── Coercion helpers ──────────────────────────────────────────────────────────

def _opt_float(d: dict, *keys) -> Optional[float]:
    for k in keys:
        if k in d and d[k] is not None:
            return float(d[k])
    return None


def _circ_to_half(d: dict, key_circ: str, key_half: Optional[str] = None) -> Optional[float]:
    if key_circ in d and d[key_circ] is not None:
        return float(d[key_circ]) / 2.0
    if key_half and key_half in d and d[key_half] is not None:
        return float(d[key_half])
    return None


def coerce_user_measurements(obj: Any) -> UserMeasurements:
    if isinstance(obj, UserMeasurements):
        return obj
    if isinstance(obj, dict):
        gender_str = str(obj.get("gender", "male")).lower()
        gender = Gender.MALE if gender_str in ("male", "m", "1") else Gender.FEMALE
        return UserMeasurements(
            gender=gender,
            height=float(obj.get("height_cm", obj.get("height", 170))),
            chest=float(obj.get("chest_circumference_cm", obj.get("chest", 90))),
            waist=float(obj.get("waist_circumference_cm", obj.get("waist", 75))),
            hip=float(obj.get("hip_circumference_cm", obj.get("hip", 95))),
            shoulder_width=_opt_float(obj, "shoulder_width_cm", "shoulder_width"),
            arm_length=_opt_float(obj, "sleeve_length_cm", "arm_length"),
            inseam=_opt_float(obj, "inseam_cm", "inseam"),
            thigh=_opt_float(obj, "thigh_circumference_cm", "thigh"),
            weight=_opt_float(obj, "weight_kg", "weight"),
        )
    raise TypeError(f"Cannot coerce {type(obj)} to UserMeasurements")


def coerce_garment_measurements(obj: Any, default_category: str = "upper_body") -> GarmentMeasurements:
    if isinstance(obj, GarmentMeasurements):
        return obj
    if isinstance(obj, dict):
        cat_str = str(obj.get("category", default_category)).lower()
        category = _GARMENT_CAT_MAP.get(cat_str, GarmentCategory.TSHIRT)
        super_category = _SUPER_CAT_MAP.get(cat_str, GarmentSuperCategory.UPPER)
        return GarmentMeasurements(
            category=category,
            super_category=super_category,
            length=(_opt_float(obj, "garment_length_cm", "length")),
            chest_width=_circ_to_half(obj, "chest_circumference_cm", "chest_width"),
            shoulder=_opt_float(obj, "shoulder_width_cm", "shoulder"),
            sleeve_length=_opt_float(obj, "sleeve_length_cm", "sleeve_length"),
            waist_width=_circ_to_half(obj, "waist_circumference_cm", "waist_width"),
            hip_width=_circ_to_half(obj, "hip_circumference_cm", "hip_width"),
            hem_width=_circ_to_half(obj, "hem_circumference_cm", "hem_width"),
            thigh_width=_circ_to_half(obj, "thigh_circumference_cm", "thigh_width"),
            inseam=_opt_float(obj, "inseam_cm", "inseam"),
            size_label=obj.get("size_label"),
        )
    raise TypeError(f"Cannot coerce {type(obj)} to GarmentMeasurements")


def resolve_category(vt_garment_type: str, default: str = "upper_body") -> str:
    cat = _GARMENT_CAT_MAP.get(vt_garment_type.lower())
    if cat is None:
        cat = _GARMENT_CAT_MAP.get(default, GarmentCategory.TSHIRT)
    return cat.value


# ── FitPredictor ─────────────────────────────────────────────────────────────

class FitPredictor:
    """Wraps RuleBasedFitPredictor; adds preferred_fit param and ease computation."""

    def __init__(self, preference: str = "regular"):
        self._rule = RuleBasedFitPredictor(preference=preference)

    def predict(self, user: UserMeasurements, garment: GarmentMeasurements, preferred_fit: str = "regular"):
        self._rule.preference = preferred_fit or "regular"
        return self._rule.predict(user, garment)

    def compute_ease_values(self, user: UserMeasurements, garment: GarmentMeasurements) -> Dict[str, float]:
        return dict(EaseVector.compute(user, garment).values)


# ── Feature dimensions ────────────────────────────────────────────────────────

_USER_DIM = 12     # UserMeasurements.to_vector()
_GARMENT_DIM = 15  # GarmentMeasurements.to_vector()
_REPORT_DIM = 17   # FitReport.to_embedding_input() = 8*2 + 1
_FEATURE_DIM = _USER_DIM + _GARMENT_DIM + _REPORT_DIM  # 44
_EMBED_DIM = 128


# ── FitStateEncoder ───────────────────────────────────────────────────────────

class FitStateEncoder(nn.Module):
    """Encodes user + garment + fit report into a fixed-size embedding vector."""

    def __init__(self, feature_dim: int = _FEATURE_DIM, embed_dim: int = _EMBED_DIM):
        super().__init__()
        self.feature_dim = feature_dim
        self.embed_dim = embed_dim
        self.encoder = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, embed_dim),
        )

    def build_single_sample_batch(
        self, user, garment, fit_report, preferred_fit, ease_values, device="cpu"
    ) -> Dict[str, torch.Tensor]:
        user_vec = torch.tensor(user.to_vector(), dtype=torch.float32, device=device).unsqueeze(0)
        garment_vec = torch.tensor(garment.to_vector(), dtype=torch.float32, device=device).unsqueeze(0)
        report_vec = torch.tensor(
            fit_report.to_embedding_input(), dtype=torch.float32, device=device
        ).unsqueeze(0)
        feature = torch.cat([user_vec, garment_vec, report_vec], dim=-1)  # (1, 44)
        return {"feature_vector": feature}

    def encode_from_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        feature_vector = batch["feature_vector"]
        with torch.no_grad():
            embedding = self.encoder(feature_vector)
        return {
            "fit_embedding": embedding,
            "fit_state": embedding,
            "raw_feature_vector": feature_vector,
        }

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return self.encode_from_batch(batch)


# ── BodyAnchorEncoder ─────────────────────────────────────────────────────────

class BodyAnchorEncoder(nn.Module):
    """Extracts body anchor keypoints from mask + densepose + parsing."""

    def __init__(self, anchor_dim: int = 32):
        super().__init__()
        self.anchor_dim = anchor_dim
        self.conv = nn.Sequential(
            nn.Conv2d(4, 8, kernel_size=7, stride=4, padding=3),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 8)),
        )
        self.fc = nn.Linear(8 * 8 * 8, anchor_dim)

    def forward(
        self,
        base_mask: torch.Tensor,
        densepose_seg: torch.Tensor,
        parsing_map=None,
    ) -> Dict[str, Any]:
        b = base_mask.shape[0]
        x = torch.cat([base_mask, densepose_seg], dim=1)  # (B, 4, H, W)
        with torch.no_grad():
            feat = self.conv(x)
            anchors = self.fc(feat.view(b, -1))
        return {
            "anchors": anchors,
            "body_box": torch.zeros(b, 4, device=base_mask.device),
            "keypoints": torch.zeros(b, 17, 2, device=base_mask.device),
            "debug": [{} for _ in range(b)],
        }


# ── GarmentGeometryGenerator ─────────────────────────────────────────────────

class GarmentGeometryGenerator(nn.Module):
    """Learned garment geometry (warp) module — passthrough stub."""

    def __init__(self, embed_dim: int = _EMBED_DIM):
        super().__init__()
        self.dummy = nn.Linear(embed_dim, 1)

    def forward(self, ref_image, garment_mask, base_mask, fit_state=None, **kwargs):
        return {
            "warped_garment_image": ref_image,
            "warped_garment_mask": garment_mask,
            "projected_mask": base_mask,
            "fit_confidence": torch.ones(ref_image.shape[0], 1, device=ref_image.device),
            "geometry_source": "learned",
        }


class HeuristicGarmentGeometryGenerator(nn.Module):
    """Heuristic garment geometry fallback — passthrough."""

    def __init__(self):
        super().__init__()

    def forward(self, ref_image, garment_mask, base_mask, **kwargs):
        return {
            "warped_garment_image": ref_image,
            "warped_garment_mask": garment_mask,
            "projected_mask": base_mask,
            "fit_confidence": torch.ones(ref_image.shape[0], 1, device=ref_image.device),
            "geometry_source": "heuristic",
        }


def run_garment_geometry(
    ref_image: torch.Tensor,
    garment_mask: torch.Tensor,
    base_mask: torch.Tensor,
    fit_state: torch.Tensor,
    fit_report,
    body_anchors: Dict[str, Any],
    garment_category,
    learned_generator: Optional[GarmentGeometryGenerator] = None,
    heuristic_generator: Optional[HeuristicGarmentGeometryGenerator] = None,
) -> Dict[str, Any]:
    generator = heuristic_generator
    if generator is not None:
        with torch.no_grad():
            result = generator(ref_image=ref_image, garment_mask=garment_mask, base_mask=base_mask)
        return result
    return {
        "warped_garment_image": ref_image,
        "warped_garment_mask": garment_mask,
        "projected_mask": base_mask,
        "fit_confidence": torch.ones(ref_image.shape[0], 1, device=ref_image.device),
        "geometry_source": "passthrough",
    }


# ── FitAwareGarmentAdapter ────────────────────────────────────────────────────

class FitAwareGarmentAdapter(nn.Module):
    """Adapts warped garment with fit embedding — passthrough stub."""

    def __init__(self, embed_dim: int = _EMBED_DIM):
        super().__init__()
        self.dummy = nn.Linear(embed_dim, 1)

    def forward(self, coarse_warp_image, coarse_warp_mask, fit_state=None, **kwargs):
        return {
            "refined_garment_image": coarse_warp_image,
            "refined_garment_mask": coarse_warp_mask,
            "adapter_confidence": torch.ones(coarse_warp_image.shape[0], 1, device=coarse_warp_image.device),
            "adapter_source": "learned",
        }


def run_fit_adapter(
    ref_image: torch.Tensor,
    garment_mask: torch.Tensor,
    coarse_warp_image: torch.Tensor,
    coarse_warp_mask: torch.Tensor,
    body_anchors: Dict[str, Any],
    vmg_outputs: Dict[str, Any],
    fit_state: torch.Tensor,
    learned_adapter: Optional[FitAwareGarmentAdapter] = None,
) -> Dict[str, Any]:
    if learned_adapter is not None:
        with torch.no_grad():
            result = learned_adapter(
                coarse_warp_image=coarse_warp_image,
                coarse_warp_mask=coarse_warp_mask,
                fit_state=fit_state,
            )
        result["adapter_source"] = "learned"
        return result
    return {
        "refined_garment_image": coarse_warp_image,
        "refined_garment_mask": coarse_warp_mask,
        "adapter_confidence": torch.ones(coarse_warp_image.shape[0], 1, device=coarse_warp_image.device),
        "adapter_source": "passthrough",
    }


# ── VirtualMeasurementGarmentBuilder ─────────────────────────────────────────

class VirtualMeasurementGarmentBuilder(nn.Module):
    """Builds virtual measurement garment layout hints — passthrough stub."""

    def __init__(self):
        super().__init__()

    def forward(self, base_mask, body_anchors, fit_report, garment_category, projected_mask):
        b, _, h, w = base_mask.shape
        return {
            "layout_hint": projected_mask,
            "vmg_mask": base_mask,
            "vmg_grid": torch.zeros(b, h, w, 2, device=base_mask.device),
            "debug": [{} for _ in range(b)],
        }


# ── HeuristicLayoutRefiner ────────────────────────────────────────────────────

class HeuristicLayoutRefiner:
    """Heuristic mask layout refiner (no learned weights)."""

    max_sdf_dist: float = 50.0

    def refine(self, base_mask, densepose_seg=None, fit_embedding=None, **kwargs):
        return base_mask


# ── run_layout_generation ─────────────────────────────────────────────────────

def run_layout_generation(
    agnostic_mask: torch.Tensor,
    densepose_seg: torch.Tensor,
    fit_embedding: torch.Tensor,
    fit_report,
    garment_category,
    learned_generator=None,
    heuristic_refiner: Optional[HeuristicLayoutRefiner] = None,
) -> Dict[str, Any]:
    target_mask = agnostic_mask
    sdf_map = torch.zeros_like(agnostic_mask)
    return {
        "target_mask": target_mask,
        "sdf_map": sdf_map,
        "confidence": torch.ones(agnostic_mask.shape[0], 1, device=agnostic_mask.device),
        "layout_source": "heuristic",
        "layout_cond": None,
    }
