import gc
import logging
import os
from typing import Any, Optional

log = logging.getLogger("capvton.model")

import cv2
import numpy as np
import torch
from PIL import Image
from diffusers import ControlNetModel, StableDiffusionControlNetInpaintPipeline

from capvton.fit import (
    BodyAnchorEncoder,
    FitEmbeddingEncoder,
    FitAwareGarmentAdapter,
    FitStateEncoder,
    GarmentGeometryGenerator,
    FitLayoutGenerator,
    FitPredictor,
    HeuristicGarmentGeometryGenerator,
    HeuristicLayoutRefiner,
    VirtualMeasurementGarmentBuilder,
    coerce_garment_measurements,
    coerce_user_measurements,
    resolve_category,
    run_fit_adapter,
    run_garment_geometry,
    run_layout_generation,
    SizeRecommender,
    RuleBasedFitPredictor,
)
from capvton.fit.config import (
    FIT_ADAPTER_CHECKPOINT_NAME,
    FIT_EMBEDDING_CHECKPOINT_NAME,
    GEOMETRY_CHECKPOINT_NAME,
    LAYOUT_CHECKPOINT_NAME,
    MASK_FUSION_DEFAULTS,
)
from capvton.fit.schema import (
    FitPreference,
    FitReport,
    GarmentCategory,
    GarmentMeasurements,
    GarmentSuperCategory,
    Gender,
    UserMeasurements,
)
from capvton.fit.utils import (
    combine_masks,
    extract_garment_foreground_mask,
    pil_mask_to_tensor,
    pil_rgb_to_tensor,
    save_fit_artifacts,
    tensor_to_list,
    tensor_to_pil_mask,
    tensor_to_pil_rgb,
)
from capvton.inference import LeffaInference
from capvton.model import LeffaModel
from capvton.transform import LeffaTransform
from capvton_utils.densepose_predictor import DensePosePredictor
from capvton_utils.garment_agnostic_mask_predictor import AutoMasker
from capvton_utils.utils import resize_and_center
from preprocess.humanparsing.run_parsing import Parsing
from preprocess.openpose.run_openpose import OpenPose


# ──────────────────────────────────────────────
# Dict → Dataclass conversion helpers
# ──────────────────────────────────────────────

_PREFERRED_FIT_MAP = {
    "slim": FitPreference.SLIM,
    "tight": FitPreference.SLIM,
    "regular": FitPreference.REGULAR,
    "normal": FitPreference.REGULAR,
    "loose": FitPreference.RELAXED,
    "relaxed": FitPreference.RELAXED,
    "oversized": FitPreference.RELAXED,
}

_GARMENT_CAT_MAP = {
    "tshirt": GarmentCategory.TSHIRT,
    "t-shirt": GarmentCategory.TSHIRT,
    "shirt": GarmentCategory.SHIRT,
    "jacket": GarmentCategory.JACKET,
    "sweater": GarmentCategory.SWEATER,
    "hoodie": GarmentCategory.HOODIE,
    "blouse": GarmentCategory.BLOUSE,
    "coat": GarmentCategory.COAT,
    "pants": GarmentCategory.PANTS,
    "jeans": GarmentCategory.JEANS,
    "skirt": GarmentCategory.SKIRT,
    "shorts": GarmentCategory.SHORTS,
    "dress": GarmentCategory.DRESS,
    "jumpsuit": GarmentCategory.JUMPSUIT,
    "upper_body": GarmentCategory.TSHIRT,
    "lower_body": GarmentCategory.PANTS,
    "dresses": GarmentCategory.DRESS,
}

_SUPER_CAT_MAP = {
    "upper_body": GarmentSuperCategory.UPPER,
    "lower_body": GarmentSuperCategory.LOWER,
    "dresses": GarmentSuperCategory.DRESS,
    "dress": GarmentSuperCategory.DRESS,
    "tshirt": GarmentSuperCategory.UPPER,
    "t-shirt": GarmentSuperCategory.UPPER,
    "shirt": GarmentSuperCategory.UPPER,
    "jacket": GarmentSuperCategory.UPPER,
    "sweater": GarmentSuperCategory.UPPER,
    "hoodie": GarmentSuperCategory.UPPER,
    "blouse": GarmentSuperCategory.UPPER,
    "coat": GarmentSuperCategory.UPPER,
    "pants": GarmentSuperCategory.LOWER,
    "jeans": GarmentSuperCategory.LOWER,
    "skirt": GarmentSuperCategory.LOWER,
    "shorts": GarmentSuperCategory.LOWER,
    "jumpsuit": GarmentSuperCategory.DRESS,
}


def _opt_float(d: dict, *keys) -> Optional[float]:
    for k in keys:
        if k in d and d[k] is not None:
            return float(d[k])
    return None


def _parse_user_measurements(d: dict) -> UserMeasurements:
    """dict 형식 → UserMeasurements 변환."""
    gender_str = str(d.get("gender", "male")).lower()
    gender = Gender.MALE if gender_str in ("male", "m", "1") else Gender.FEMALE
    return UserMeasurements(
        gender=gender,
        height=float(d.get("height_cm", d.get("height", 170))),
        chest=float(d.get("chest_circumference_cm", d.get("chest", 90))),
        waist=float(d.get("waist_circumference_cm", d.get("waist", 75))),
        hip=float(d.get("hip_circumference_cm", d.get("hip", 95))),
        shoulder_width=_opt_float(d, "shoulder_width_cm", "shoulder_width"),
        arm_length=_opt_float(d, "sleeve_length_cm", "arm_length"),
        inseam=_opt_float(d, "inseam_cm", "inseam"),
        thigh=_opt_float(d, "thigh_circumference_cm", "thigh"),
        weight=_opt_float(d, "weight_kg", "weight"),
    )


def _parse_garment_measurements(d: dict) -> GarmentMeasurements:
    """dict 형식 → GarmentMeasurements 변환."""
    cat_str = str(d.get("category", "upper_body")).lower()
    category = _GARMENT_CAT_MAP.get(cat_str, GarmentCategory.TSHIRT)
    super_category = _SUPER_CAT_MAP.get(cat_str, GarmentSuperCategory.UPPER)

    def _circ_to_half(key_circ, key_half=None):
        if key_circ in d:
            return float(d[key_circ]) / 2.0
        if key_half and key_half in d:
            return float(d[key_half])
        return None

    return GarmentMeasurements(
        category=category,
        super_category=super_category,
        length=float(d.get("garment_length_cm", d.get("length", 65))),
        chest_width=_circ_to_half("chest_circumference_cm", "chest_width"),
        shoulder=_opt_float(d, "shoulder_width_cm", "shoulder"),
        sleeve_length=_opt_float(d, "sleeve_length_cm", "sleeve_length"),
        waist_width=_circ_to_half("waist_circumference_cm", "waist_width"),
        hip_width=_circ_to_half("hip_circumference_cm", "hip_width"),
        hem_width=_circ_to_half("hem_circumference_cm", "hem_width"),
        thigh_width=_circ_to_half("thigh_circumference_cm", "thigh_width"),
        inseam=_opt_float(d, "inseam_cm", "inseam"),
        size_label=d.get("size_label"),
    )


class CAPVirtualTryOn:
    def __init__(self, ckpt_dir: str):
        self.ckpt_dir = ckpt_dir

        # --- Lightweight preprocessing models ---
        self.mask_predictor = AutoMasker(
            densepose_path=f"{ckpt_dir}/densepose",
            schp_path=f"{ckpt_dir}/schp",
        )

        # Reuse detectron2 predictor from AutoMasker (~170MB GPU saved)
        self.densepose_predictor = DensePosePredictor(
            predictor=self.mask_predictor.densepose_processor.predictor,
        )

        # Human parsing: ONNX on CPU — no GPU cost
        self.parsing = Parsing(
            atr_path=f"{ckpt_dir}/humanparsing/parsing_atr.onnx",
            lip_path=f"{ckpt_dir}/humanparsing/parsing_lip.onnx",
        )

        self.openpose = OpenPose(
            body_model_path=f"{ckpt_dir}/openpose/body_pose_model.pth",
        )

        # --- Fit-aware lightweight modules ---
        self.fit_predictor = FitPredictor()
        self.fit_state_encoder = FitStateEncoder().to("cpu")
        self.fit_embedding_encoder = self.fit_state_encoder
        self.body_anchor_encoder = BodyAnchorEncoder().to("cpu")
        self.heuristic_garment_geometry = HeuristicGarmentGeometryGenerator().to("cpu")
        self._garment_geometry_generator = None
        self.virtual_measurement_builder = VirtualMeasurementGarmentBuilder().to("cpu")
        self._fit_adapter = None
        self.heuristic_layout_refiner = HeuristicLayoutRefiner()
        self._fit_layout_generator = None
        self._fit_embedding_weights_loaded = False
        self.size_recommender = SizeRecommender(self.fit_predictor)
        self._rule_fit_predictor = RuleBasedFitPredictor(preference="regular")

        # --- Heavy diffusion models: lazy-loaded, only one on GPU at a time ---
        self._vt_inference_hd = None
        self._vt_inference_dc = None
        self._skin_pipe = None

    # ------------------------------------------------------------------
    # GPU memory management
    # ------------------------------------------------------------------
    def _free_gpu(self):
        """Remove heavy diffusion models from GPU."""
        if self._skin_pipe is not None:
            del self._skin_pipe
            self._skin_pipe = None
        if self._vt_inference_hd is not None:
            self._vt_inference_hd.model.cpu()
        if self._vt_inference_dc is not None:
            self._vt_inference_dc.model.cpu()
        gc.collect()
        torch.cuda.empty_cache()

    def _offload_preprocessing(self):
        """Move all preprocessing GPU models to CPU (~880MB freed)."""
        for accessor in [
            lambda: self.mask_predictor.schp_processor_atr.model,
            lambda: self.mask_predictor.schp_processor_lip.model,
            lambda: self.mask_predictor.densepose_processor.predictor.model,
            lambda: self.openpose.preprocessor.body_estimation.model,
        ]:
            try:
                accessor().cpu()
            except (AttributeError, RuntimeError):
                pass
        gc.collect()
        torch.cuda.empty_cache()

    def _ensure_preprocessing_on_gpu(self):
        """Reload preprocessing models to GPU (reverses _offload_preprocessing)."""
        for accessor in [
            lambda: self.mask_predictor.schp_processor_atr.model,
            lambda: self.mask_predictor.schp_processor_lip.model,
            lambda: self.mask_predictor.densepose_processor.predictor.model,
            lambda: self.openpose.preprocessor.body_estimation.model,
        ]:
            try:
                accessor().cuda()
            except (AttributeError, RuntimeError):
                pass

    def _get_skin_pipe(self):
        """Lazy-load skin inpainting pipeline onto GPU."""
        self._free_gpu()
        if self._skin_pipe is None:
            controlnet = ControlNetModel.from_pretrained(
                "lllyasviel/sd-controlnet-openpose",
                torch_dtype=torch.float16,
            )
            self._skin_pipe = StableDiffusionControlNetInpaintPipeline.from_single_file(
                f"{self.ckpt_dir}/majicmixRealistic_v7.safetensors",
                controlnet=controlnet,
                torch_dtype=torch.float16,
                safety_checker=None,
            )
        self._skin_pipe = self._skin_pipe.to("cuda")
        return self._skin_pipe

    def _get_vt_inference(self, model_type: str):
        """Lazy-load virtual try-on model onto GPU."""
        self._free_gpu()
        if model_type == "viton_hd":
            if self._vt_inference_hd is None:
                vt_model = LeffaModel(
                    pretrained_model_name_or_path=f"{self.ckpt_dir}/stable-diffusion-inpainting",
                    pretrained_model=f"{self.ckpt_dir}/virtual_tryon.pth",
                    dtype="float16",
                )
                self._vt_inference_hd = LeffaInference(model=vt_model, auto_device=False)
            self._vt_inference_hd.model.to("cuda")
            return self._vt_inference_hd

        if self._vt_inference_dc is None:
            vt_model = LeffaModel(
                pretrained_model_name_or_path=f"{self.ckpt_dir}/stable-diffusion-inpainting",
                pretrained_model=f"{self.ckpt_dir}/virtual_tryon_dc.pth",
                dtype="float16",
            )
            self._vt_inference_dc = LeffaInference(model=vt_model, auto_device=False)
        self._vt_inference_dc.model.to("cuda")
        return self._vt_inference_dc

    # ------------------------------------------------------------------
    # Fit-aware helpers
    # ------------------------------------------------------------------
    def _fit_checkpoint_dir(self) -> str:
        return os.path.join(self.ckpt_dir, "fit")

    def _fit_embedding_checkpoint_path(self) -> str:
        return os.path.join(self._fit_checkpoint_dir(), FIT_EMBEDDING_CHECKPOINT_NAME)

    def _geometry_checkpoint_path(self) -> str:
        return os.path.join(self._fit_checkpoint_dir(), GEOMETRY_CHECKPOINT_NAME)

    def _fit_adapter_checkpoint_path(self) -> str:
        return os.path.join(self._fit_checkpoint_dir(), FIT_ADAPTER_CHECKPOINT_NAME)

    def _layout_generator_checkpoint_path(self) -> str:
        return os.path.join(self._fit_checkpoint_dir(), LAYOUT_CHECKPOINT_NAME)

    def _get_fit_state_encoder(self):
        encoder = getattr(self, "fit_state_encoder", None)
        if encoder is None:
            encoder = getattr(self, "fit_embedding_encoder", None)
        if encoder is None:
            encoder = FitStateEncoder().to("cpu")
            self.fit_state_encoder = encoder
            self.fit_embedding_encoder = encoder
        return encoder

    def _get_body_anchor_encoder(self):
        encoder = getattr(self, "body_anchor_encoder", None)
        if encoder is None:
            encoder = BodyAnchorEncoder().to("cpu")
            self.body_anchor_encoder = encoder
        return encoder

    def _get_heuristic_garment_geometry(self):
        generator = getattr(self, "heuristic_garment_geometry", None)
        if generator is None:
            generator = HeuristicGarmentGeometryGenerator().to("cpu")
            self.heuristic_garment_geometry = generator
        return generator

    def _get_virtual_measurement_builder(self):
        builder = getattr(self, "virtual_measurement_builder", None)
        if builder is None:
            builder = VirtualMeasurementGarmentBuilder().to("cpu")
            self.virtual_measurement_builder = builder
        return builder

    def _ensure_fit_embedding_weights(self) -> None:
        if self._fit_embedding_weights_loaded:
            return

        checkpoint_path = self._fit_embedding_checkpoint_path()
        state_encoder = self._get_fit_state_encoder()
        if os.path.exists(checkpoint_path):
            state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            if isinstance(state_dict, dict) and "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            state_encoder.load_state_dict(state_dict, strict=False)
        state_encoder.eval()
        self.fit_state_encoder = state_encoder
        self.fit_embedding_encoder = state_encoder
        self._fit_embedding_weights_loaded = True

    def _get_garment_geometry_generator(self) -> Optional[GarmentGeometryGenerator]:
        checkpoint_path = self._geometry_checkpoint_path()
        if not os.path.exists(checkpoint_path):
            return None
        if getattr(self, "_garment_geometry_generator", None) is None:
            self._garment_geometry_generator = GarmentGeometryGenerator().to("cpu")
            state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            if isinstance(state_dict, dict) and "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            self._garment_geometry_generator.load_state_dict(state_dict, strict=False)
            self._garment_geometry_generator.eval()
        return self._garment_geometry_generator

    def _get_fit_adapter(self) -> Optional[FitAwareGarmentAdapter]:
        checkpoint_path = self._fit_adapter_checkpoint_path()
        if not os.path.exists(checkpoint_path):
            return None
        if getattr(self, "_fit_adapter", None) is None:
            self._fit_adapter = FitAwareGarmentAdapter().to("cpu")
            state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            if isinstance(state_dict, dict) and "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            self._fit_adapter.load_state_dict(state_dict, strict=False)
            self._fit_adapter.eval()
        return self._fit_adapter

    def _get_fit_layout_generator(self) -> Optional[FitLayoutGenerator]:
        checkpoint_path = self._layout_generator_checkpoint_path()
        if not os.path.exists(checkpoint_path):
            return None
        if self._fit_layout_generator is None:
            self._fit_layout_generator = FitLayoutGenerator().to("cpu")
            state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            if isinstance(state_dict, dict) and "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            self._fit_layout_generator.load_state_dict(state_dict, strict=False)
            self._fit_layout_generator.eval()
        return self._fit_layout_generator

    def _fit_mode_requested(
        self,
        user_measurements: Any,
        garment_measurements: Any,
        control_type: str,
    ) -> bool:
        return (
            control_type == "virtual_tryon"
            and user_measurements is not None
            and garment_measurements is not None
        )

    def _run_fit_aware_geometry(
        self,
        base_mask: Image.Image,
        densepose: Image.Image,
        parsing_map: np.ndarray,
        ref_image: Image.Image,
        user_measurements: Any,
        garment_measurements: Any,
        preferred_fit: Optional[str],
        vt_garment_type: str,
    ) -> dict[str, Any]:
        default_category = resolve_category(vt_garment_type, default="upper_body")
        user = coerce_user_measurements(user_measurements)
        garment = coerce_garment_measurements(
            garment_measurements,
            default_category=default_category,
        )
        preferred_fit = (preferred_fit or "regular").lower()

        fit_report = self.fit_predictor.predict(user, garment, preferred_fit=preferred_fit)
        ease_values = self.fit_predictor.compute_ease_values(user, garment)

        self._ensure_fit_embedding_weights()
        fit_state_encoder = self._get_fit_state_encoder()
        with torch.no_grad():
            batch = fit_state_encoder.build_single_sample_batch(
                user=user,
                garment=garment,
                fit_report=fit_report,
                preferred_fit=preferred_fit,
                ease_values=ease_values,
                device="cpu",
            )
            fit_state_outputs = fit_state_encoder.encode_from_batch(batch)
            fit_state = fit_state_outputs.get("fit_state", fit_state_outputs["fit_embedding"])
            raw_feature_vector = fit_state_outputs["raw_feature_vector"]

        base_mask_tensor = pil_mask_to_tensor(base_mask)
        densepose_tensor = pil_rgb_to_tensor(densepose)
        ref_image_tensor = pil_rgb_to_tensor(ref_image)
        garment_mask_image = extract_garment_foreground_mask(ref_image)
        garment_mask_tensor = pil_mask_to_tensor(garment_mask_image)

        body_anchor_encoder = self._get_body_anchor_encoder()
        with torch.no_grad():
            body_anchor_outputs = body_anchor_encoder(
                base_mask=base_mask_tensor,
                densepose_seg=densepose_tensor,
                parsing_map=parsing_map,
            )

        geometry_result = run_garment_geometry(
            ref_image=ref_image_tensor,
            garment_mask=garment_mask_tensor,
            base_mask=base_mask_tensor,
            fit_state=fit_state,
            fit_report=fit_report,
            body_anchors=body_anchor_outputs,
            garment_category=garment.category,
            learned_generator=self._get_garment_geometry_generator(),
            heuristic_generator=self._get_heuristic_garment_geometry(),
        )

        conservative_mask_tensor = (base_mask_tensor >= 0.5).float()
        warped_garment_image_tensor = geometry_result["warped_garment_image"]
        warped_garment_mask_tensor = geometry_result["warped_garment_mask"]
        projected_mask_tensor = geometry_result["projected_mask"]
        vmg_outputs = self._get_virtual_measurement_builder()(
            base_mask=base_mask_tensor,
            body_anchors=body_anchor_outputs,
            fit_report=fit_report,
            garment_category=garment.category,
            projected_mask=projected_mask_tensor,
        )
        adapter_result = run_fit_adapter(
            ref_image=ref_image_tensor,
            garment_mask=garment_mask_tensor,
            coarse_warp_image=warped_garment_image_tensor,
            coarse_warp_mask=warped_garment_mask_tensor,
            body_anchors=body_anchor_outputs,
            vmg_outputs=vmg_outputs,
            fit_state=fit_state,
            learned_adapter=self._get_fit_adapter(),
        )
        adapted_garment_image_tensor = adapter_result["refined_garment_image"]
        adapted_garment_mask_tensor = adapter_result["refined_garment_mask"]
        warped_garment_image_tensor = adapted_garment_image_tensor
        warped_garment_mask_tensor = adapted_garment_mask_tensor
        layout_hint_tensor = vmg_outputs["layout_hint"]
        vmg_mask_tensor = vmg_outputs["vmg_mask"]
        vmg_grid_tensor = vmg_outputs["vmg_grid"]

        warped_garment_image = tensor_to_pil_rgb(warped_garment_image_tensor)
        warped_garment_mask = tensor_to_pil_mask(warped_garment_mask_tensor)
        layout_hint = tensor_to_pil_mask(projected_mask_tensor)
        conservative_mask = tensor_to_pil_mask(conservative_mask_tensor)
        vmg_mask = tensor_to_pil_mask(vmg_mask_tensor)

        fit_labels = {part_name: part.label for part_name, part in fit_report.parts.items()}
        base_mask_area = float((base_mask_tensor >= 0.5).float().mean().item())
        warped_mask_area = float((warped_garment_mask_tensor >= 0.5).float().mean().item())

        geometry_source = geometry_result.get("geometry_source", "heuristic")
        geometry_confidence = float(geometry_result["fit_confidence"].detach().cpu().flatten()[0].item())
        adapter_confidence = float(adapter_result["adapter_confidence"].detach().cpu().flatten()[0].item())
        debug = {
            "base_mask": base_mask,
            "layout_source": geometry_source,
            "geometry_source": geometry_source,
            "geometry_confidence": geometry_confidence,
            "adapter_source": adapter_result.get("adapter_source", "disabled"),
            "adapter_confidence": adapter_confidence,
            "fit_state": tensor_to_list(fit_state[0]),
            "raw_feature_vector": tensor_to_list(raw_feature_vector[0]),
            "body_anchors": body_anchor_outputs["debug"][0],
            "vmg_debug": vmg_outputs["debug"][0],
            "ease_values": ease_values,
            "fit_labels": fit_labels,
            "user_measurements": user.to_dict(),
            "garment_measurements": garment.to_dict(),
            "preferred_fit": preferred_fit,
            "base_mask_area": base_mask_area,
            "warped_mask_area": warped_mask_area,
            "fit_strategy": "geometry",
            "masked_image_mask_source": "base_mask",
        }

        return {
            "fit_report": fit_report,
            "fit_state": fit_state,
            "warped_garment_image": warped_garment_image,
            "warped_garment_image_tensor": warped_garment_image_tensor,
            "warped_garment_mask": warped_garment_mask,
            "warped_garment_mask_tensor": warped_garment_mask_tensor,
            "layout_hint": layout_hint,
            "layout_hint_tensor": layout_hint_tensor,
            "projected_mask_tensor": projected_mask_tensor,
            "vmg_mask": vmg_mask,
            "vmg_mask_tensor": vmg_mask_tensor,
            "vmg_grid_tensor": vmg_grid_tensor,
            "adapted_garment_image_tensor": adapted_garment_image_tensor,
            "adapted_garment_mask_tensor": adapted_garment_mask_tensor,
            "base_mask": base_mask,
            "base_mask_tensor": base_mask_tensor,
            "refined_mask": conservative_mask,
            "refined_mask_tensor": conservative_mask_tensor,
            "masked_image_mask": base_mask,
            "masked_image_mask_tensor": conservative_mask_tensor,
            "debug": debug,
            "fit_strategy": "geometry",
        }

    def _run_fit_aware_refinement(
        self,
        base_mask: Image.Image,
        densepose: Image.Image,
        user_measurements: Any,
        garment_measurements: Any,
        preferred_fit: Optional[str],
        vt_garment_type: str,
    ) -> dict[str, Any]:
        default_category = resolve_category(vt_garment_type, default="upper_body")
        user = coerce_user_measurements(user_measurements)
        garment = coerce_garment_measurements(
            garment_measurements,
            default_category=default_category,
        )
        preferred_fit = (preferred_fit or "regular").lower()

        fit_report = self.fit_predictor.predict(user, garment, preferred_fit=preferred_fit)
        ease_values = self.fit_predictor.compute_ease_values(user, garment)

        self._ensure_fit_embedding_weights()
        fit_state_encoder = self._get_fit_state_encoder()
        with torch.no_grad():
            batch = fit_state_encoder.build_single_sample_batch(
                user=user,
                garment=garment,
                fit_report=fit_report,
                preferred_fit=preferred_fit,
                ease_values=ease_values,
                device="cpu",
            )
            embedding_outputs = fit_state_encoder.encode_from_batch(batch)
            fit_embedding = embedding_outputs["fit_embedding"]
            raw_feature_vector = embedding_outputs["raw_feature_vector"]

        base_mask_tensor = pil_mask_to_tensor(base_mask)
        densepose_tensor = pil_rgb_to_tensor(densepose)
        layout_result = run_layout_generation(
            agnostic_mask=base_mask_tensor,
            densepose_seg=densepose_tensor,
            fit_embedding=fit_embedding,
            fit_report=fit_report,
            garment_category=garment.category,
            learned_generator=self._get_fit_layout_generator(),
            heuristic_refiner=self.heuristic_layout_refiner,
        )

        target_mask_tensor = layout_result["target_mask"]
        sdf_map_tensor = layout_result["sdf_map"]
        refined_mask_tensor = combine_masks(
            base_mask_tensor,
            target_mask_tensor,
            mode=MASK_FUSION_DEFAULTS["mode"],
            alpha=MASK_FUSION_DEFAULTS["alpha"],
            threshold=MASK_FUSION_DEFAULTS["threshold"],
        )

        target_mask = tensor_to_pil_mask(target_mask_tensor)
        refined_mask = tensor_to_pil_mask(refined_mask_tensor)
        sdf_visual_tensor = torch.clamp(
            (sdf_map_tensor / (self.heuristic_layout_refiner.max_sdf_dist * 2.0)) + 0.5,
            0.0,
            1.0,
        )
        sdf_map = tensor_to_pil_mask(sdf_visual_tensor)
        fit_labels = {part_name: part.label for part_name, part in fit_report.parts.items()}
        base_mask_area = float((base_mask_tensor >= 0.5).float().mean().item())
        target_mask_area = float((target_mask_tensor >= 0.5).float().mean().item())
        refined_mask_area = float((refined_mask_tensor >= 0.5).float().mean().item())

        debug = {
            "base_mask": base_mask,
            "layout_source": layout_result.get("layout_source", "heuristic"),
            "layout_confidence": float(layout_result["confidence"].detach().cpu().flatten()[0].item()),
            "fit_embedding": tensor_to_list(fit_embedding[0]),
            "raw_feature_vector": tensor_to_list(raw_feature_vector[0]),
            "ease_values": ease_values,
            "user_measurements": user.to_dict(),
            "garment_measurements": garment.to_dict(),
            "preferred_fit": preferred_fit,
            "fit_labels": fit_labels,
            "base_mask_area": base_mask_area,
            "target_mask_area": target_mask_area,
            "refined_mask_area": refined_mask_area,
            "masked_image_mask_source": "base_mask",
            "fit_strategy": "mask",
        }

        return {
            "fit_report": fit_report,
            "refined_mask": refined_mask,
            "refined_mask_tensor": refined_mask_tensor,
            "target_mask": target_mask,
            "target_mask_tensor": target_mask_tensor,
            "sdf_map": sdf_map,
            "sdf_map_tensor": sdf_map_tensor,
            "layout_cond": layout_result["layout_cond"],
            "debug": debug,
            "fit_strategy": "mask",
            "base_mask": base_mask,
            "base_mask_tensor": base_mask_tensor,
        }

    # ------------------------------------------------------------------
    # Fit-aware prompt and tightness helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _compute_avg_tightness(fit_report) -> float:
        if fit_report is None or not fit_report.parts:
            return 0.0
        return float(np.mean([p.tightness for p in fit_report.parts.values()]))

    def _adjust_mask_for_fit(
        self,
        mask: Image.Image,
        fit_report,
    ) -> Image.Image:
        """Erode mask for tight fit (body contour), dilate for oversized."""
        if fit_report is None or not fit_report.parts:
            return mask
        avg_tightness = self._compute_avg_tightness(fit_report)
        if abs(avg_tightness) < 0.15:
            return mask

        mask_np = np.array(mask.convert("L"))
        if avg_tightness < -0.15:
            # Tight → erode mask so garment is forced into a smaller region
            kernel_size = max(3, int(abs(avg_tightness) * 35))
            if kernel_size % 2 == 0:
                kernel_size += 1
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (kernel_size, kernel_size),
            )
            mask_np = cv2.erode(mask_np, kernel, iterations=1)
        else:
            # Oversized → dilate mask so garment can occupy a wider region
            kernel_size = max(3, int(avg_tightness * 45))
            if kernel_size % 2 == 0:
                kernel_size += 1
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (kernel_size, kernel_size),
            )
            mask_np = cv2.dilate(mask_np, kernel, iterations=1)
        return Image.fromarray(mask_np)

    def _apply_fit_post_processing(
        self,
        gen_image: Image.Image,
        skin_image: Image.Image,
        densepose: Image.Image,
        mask: Image.Image,
        fit_report,
    ) -> Image.Image:
        """Apply subtle visual cues that reinforce the predicted fit."""
        if fit_report is None or not fit_report.parts:
            return gen_image
        avg_tightness = self._compute_avg_tightness(fit_report)

        gen_np = np.array(gen_image).astype(np.float32)
        mask_np = np.array(mask.convert("L")).astype(np.float32) / 255.0

        if avg_tightness < -0.25:
            # ── Tight-fit: body contour bleed-through ──
            # Extract body edges from densepose and overlay subtle shading
            dp_gray = np.array(densepose.convert("L"))
            edges = cv2.Canny(dp_gray, 40, 120)
            edges = cv2.GaussianBlur(edges.astype(np.float32), (7, 7), 2.5)
            edge_strength = np.clip(edges / 255.0, 0, 1)
            # Blend strength proportional to how tight the fit is
            blend = min(0.30, abs(avg_tightness) * 0.35)
            shadow = edge_strength[:, :, np.newaxis] * blend * mask_np[:, :, np.newaxis]
            gen_np = gen_np * (1.0 - shadow * 0.4)  # darken along contours

        elif avg_tightness > 0.25:
            # ── Oversized: soft drape effect on lower portion ──
            h = gen_np.shape[0]
            drape_start = int(h * 0.70)
            drape_strength = min(0.20, avg_tightness * 0.18)
            if drape_start < h:
                gradient = np.zeros((h, 1, 1), dtype=np.float32)
                ramp = np.linspace(0, drape_strength, h - drape_start)
                gradient[drape_start:, :, :] = ramp[:, np.newaxis, np.newaxis]
                region_mask = mask_np[:, :, np.newaxis] * gradient
                blur_k = max(3, int(avg_tightness * 5)) | 1  # ensure odd
                blurred = cv2.GaussianBlur(gen_np, (blur_k, blur_k), 0)
                gen_np = gen_np * (1.0 - region_mask) + blurred * region_mask

        return Image.fromarray(np.clip(gen_np, 0, 255).astype(np.uint8))

    @staticmethod
    def _build_fit_prompts(fit_outputs) -> tuple[str, str]:
        base_pos = "High quality skin, lifelike details, realistic textures"
        base_neg = "distorted, blurry, low quality, artifact, background"

        if fit_outputs is None or fit_outputs.get("fit_report") is None:
            return (
                f"{base_pos}, full masking range",
                f"{base_neg}, clothes",
            )

        report = fit_outputs["fit_report"]
        if not report.parts:
            return (
                f"{base_pos}, full masking range",
                f"{base_neg}, clothes",
            )

        # Determine dominant fit label
        label_counts: dict[str, int] = {}
        for part in report.parts.values():
            label_counts[part.label] = label_counts.get(part.label, 0) + 1
        dominant = max(label_counts, key=label_counts.get)

        if dominant == "very_tight":
            prompt = (
                f"{base_pos}, extremely tight fitting garment stretched "
                "taut against body, visible body contours through fabric, "
                "skin tight, tension wrinkles pulling across surface"
            )
            neg = f"{base_neg}, loose fabric, baggy, oversized, wrinkled drape"
        elif dominant == "tight":
            prompt = (
                f"{base_pos}, snug fitted garment hugging body closely, "
                "slight fabric tension, body-conforming fit"
            )
            neg = f"{base_neg}, loose fabric, baggy, oversized"
        elif dominant == "loose":
            prompt = (
                f"{base_pos}, relaxed loose fit with extra room, "
                "soft fabric draping naturally, casual comfortable style"
            )
            neg = f"{base_neg}, skin tight, stretched, body contour"
        elif dominant == "oversized":
            prompt = (
                f"{base_pos}, very oversized baggy garment, excessive fabric "
                "draping and pooling, boxy loose silhouette, fabric bunching"
            )
            neg = f"{base_neg}, skin tight, stretched, fitted, body contour"
        else:  # regular
            prompt = (
                f"{base_pos}, well-fitted garment with comfortable "
                "regular fit, clean drape"
            )
            neg = f"{base_neg}, clothes"

        return prompt, neg

    def _format_outputs(
        self,
        generated_image: Image.Image,
        mask: Image.Image,
        densepose: Image.Image,
        agnostic_image: Image.Image,
        fit_outputs: Optional[dict[str, Any]] = None,
        return_fit_debug: bool = False,
    ):
        if fit_outputs is None and not return_fit_debug:
            return generated_image, mask, densepose, agnostic_image

        refined_mask = fit_outputs["refined_mask"] if fit_outputs else mask
        base_mask = fit_outputs.get("base_mask", mask) if fit_outputs else mask
        return {
            "generated_image": generated_image,
            "refined_mask": refined_mask,
            "mask": refined_mask,
            "base_mask": base_mask,
            "densepose": densepose,
            "agnostic_image": agnostic_image,
            "fit_report": fit_outputs["fit_report"] if fit_outputs else None,
            "target_mask": fit_outputs.get("target_mask") if fit_outputs else None,
            "sdf_map": fit_outputs.get("sdf_map") if fit_outputs else None,
            "warped_garment_image": fit_outputs.get("warped_garment_image") if fit_outputs else None,
            "warped_garment_mask": fit_outputs.get("warped_garment_mask") if fit_outputs else None,
            "virtual_measurement_garment": fit_outputs.get("vmg_mask") if fit_outputs else None,
            "layout_hint": fit_outputs.get("layout_hint") if fit_outputs else None,
            "debug": fit_outputs["debug"] if fit_outputs else {
                "base_mask": mask,
                "layout_source": "disabled",
                "fit_strategy": "disabled",
            },
        }

    # ------------------------------------------------------------------
    # Rule-based fit prediction (standalone — no GPU needed)
    # ------------------------------------------------------------------
    def fit_predict(
        self,
        user_measurements: UserMeasurements,
        garment_measurements: GarmentMeasurements,
        preference: FitPreference = FitPreference.REGULAR,
    ) -> FitReport:
        """단일 사이즈 핏 판정 (이미지 생성 없이 CPU만 사용).

        Args:
            user_measurements: 사용자 신체 치수 (UserMeasurements 또는 dict)
            garment_measurements: 의류 실측 스펙 (GarmentMeasurements 또는 dict)
            preference: 선호 핏 (FitPreference enum 또는 str)

        Returns:
            FitReport: overall_score, 부위별 tightness/fit_class/risk, notes
        """
        if isinstance(user_measurements, dict):
            user_measurements = _parse_user_measurements(user_measurements)
        if isinstance(garment_measurements, dict):
            garment_measurements = _parse_garment_measurements(garment_measurements)
        pref_str = preference.value if isinstance(preference, FitPreference) else str(preference)
        self._rule_fit_predictor.preference = pref_str
        return self._rule_fit_predictor.predict(user_measurements, garment_measurements)

    def fit_recommend_size(
        self,
        user_measurements: UserMeasurements,
        garment_sizes: dict,
        preference: FitPreference = FitPreference.REGULAR,
    ) -> tuple:
        """여러 사이즈 중 최적 사이즈 추천 (이미지 생성 없이 CPU만 사용).

        Args:
            user_measurements: 사용자 신체 치수 (UserMeasurements 또는 dict)
            garment_sizes: {"S": GarmentMeasurements, "M": ..., "L": ...}
                           값은 GarmentMeasurements 객체 또는 dict 모두 허용
            preference: 선호 핏 (FitPreference enum 또는 str)

        Returns:
            tuple: (best_label, best_report, all_reports)
        """
        if isinstance(user_measurements, dict):
            user_measurements = _parse_user_measurements(user_measurements)
        garment_sizes = {
            k: (_parse_garment_measurements(v) if isinstance(v, dict) else v)
            for k, v in garment_sizes.items()
        }
        pref_str = preference.value if isinstance(preference, FitPreference) else str(preference)
        self._rule_fit_predictor.preference = pref_str
        return self._rule_fit_predictor.recommend_size(user_measurements, garment_sizes)

    @staticmethod
    def print_fit_report(report: FitReport) -> None:
        """FitReport를 보기 좋게 출력."""
        print(f"\n{'='*50}")
        print(f"  사이즈 추천: {report.size_recommendation}")
        print(f"  전체 적합도: {report.overall_score:.0%}")
        if report.all_sizes_scores:
            scores_str = ", ".join(
                f"{s}: {v:.0%}" for s, v in report.all_sizes_scores.items()
            )
            print(f"  사이즈별 점수: {scores_str}")
        print(f"{'='*50}")
        print(f"  {'부위':<15} {'핏 클래스':<12} {'타이트니스':>10} {'리스크':<8}")
        print(f"  {'-'*45}")
        for part, r in report.parts.items():
            risk_icon = {"ok": "🟢", "caution": "🟡", "risk": "🔴"}.get(
                r.risk_level.value if hasattr(r.risk_level, "value") else str(r.risk_level), "⚪"
            )
            fit_label = r.label if hasattr(r, "label") else (
                r.fit_class.value if hasattr(r, "fit_class") else str(r)
            )
            print(f"  {part:<15} {fit_label:<12} {r.tightness:>+8.2f}   {risk_icon}")
        print(f"{'='*50}")
        for note in report.notes:
            print(f"  {note}")
        print()

    @staticmethod
    def _print_size_info(report: FitReport, garment: GarmentMeasurements) -> None:
        """사이즈 상세 정보를 출력."""
        size_label = garment.size_label if hasattr(garment, "size_label") and garment.size_label else "—"
        print(f"\n{'─'*50}")
        print(f"  [사이즈 정보] 의류 라벨: {size_label}")
        print(f"  추천 판정: {report.size_recommendation}")
        print(f"  종합 적합도: {report.overall_score:.1%}")
        if report.all_sizes_scores:
            best = max(report.all_sizes_scores, key=report.all_sizes_scores.get)
            print(f"  사이즈별 점수:")
            for sz, sc in sorted(report.all_sizes_scores.items(), key=lambda x: -x[1]):
                marker = " ◀ 최적" if sz == best else ""
                print(f"    {sz:<6} {sc:.1%}{marker}")
        if report.risk_parts:
            print(f"  주의 부위: {', '.join(report.risk_parts)}")
        else:
            print(f"  주의 부위: 없음")
        print(f"{'─'*50}\n")

    # ------------------------------------------------------------------
    # Skin inpainting (Stage 1)
    # ------------------------------------------------------------------
    def generate_skin(
        self,
        src_image: Image.Image,
        inpaint_mask_img: Image.Image,
        step: int = 20,
        seed: int = 42,
    ) -> Image.Image:
        """Inpaint realistic skin in the masked area."""
        skin_prompt = (
            "Wearing Held Tight Short Sleeve Shirt, high quality skin, realistic, high quality"
        )
        negative_prompt = (
            "Blurry, low quality, artifacts, deformed, ugly, texture, "
            "watermark, text, bad anatomy, extra limbs, face, hands, fingers"
        )

        openpose_result = self.openpose(src_image)
        openpose_image = (
            openpose_result.get("image") if isinstance(openpose_result, dict) else openpose_result
        )
        if not isinstance(openpose_image, Image.Image):
            raise TypeError(f"Invalid OpenPose output: {type(openpose_image)}")

        self._offload_preprocessing()

        generator = torch.Generator(device="cuda").manual_seed(seed)
        skin_pipe = self._get_skin_pipe()

        generated_image = skin_pipe(
            prompt=skin_prompt,
            negative_prompt=negative_prompt,
            image=src_image,
            mask_image=inpaint_mask_img,
            control_image=openpose_image,
            width=src_image.width,
            height=src_image.height,
            num_inference_steps=step,
            generator=generator,
            guidance_scale=7.0,
        ).images[0]

        src_np = np.array(src_image)
        mask_np = np.array(inpaint_mask_img.convert("L"), dtype=np.float32) / 255.0
        mask_np = mask_np[:, :, np.newaxis]
        generated_np = np.array(generated_image)
        final_np = src_np * (1.0 - mask_np) + generated_np * mask_np
        return Image.fromarray(final_np.astype(np.uint8))

    # ------------------------------------------------------------------
    # Main prediction entry-point
    # ------------------------------------------------------------------
    def capvton_predict(
        self,
        src_image_path,
        ref_image_path,
        control_type,
        ref_acceleration=False,
        output_path: str = None,
        step=20,
        cross_attention_kwargs=None,
        seed=42,
        vt_model_type="viton_hd",
        vt_garment_type="upper_body",
        vt_repaint=False,
        src_mask_path=None,
        user_measurements=None,
        garment_measurements=None,
        preferred_fit: Optional[str] = None,
        fit_preference: FitPreference = FitPreference.REGULAR,
        fit_strategy: Optional[str] = None,
        return_fit_debug: bool = False,
        **kwargs,
    ):
        assert control_type in [
            "virtual_tryon",
            "pose_transfer",
        ], f"Invalid control type: {control_type}"

        if cross_attention_kwargs is None:
            cross_attention_kwargs = {"scale": 3}

        # ---- preferred_fit(str) → fit_preference(enum) 변환 ----
        if preferred_fit is not None:
            fit_preference = _PREFERRED_FIT_MAP.get(
                str(preferred_fit).lower(), FitPreference.REGULAR
            )

        # ---- dict 입력 → dataclass 변환 ----
        if isinstance(user_measurements, dict):
            user_measurements = _parse_user_measurements(user_measurements)
        if isinstance(garment_measurements, dict):
            garment_measurements = _parse_garment_measurements(garment_measurements)

        # ---- Step 0: Rule-based 핏 판정 (GPU 불필요, 빠른 미리보기) ----
        if user_measurements is not None and garment_measurements is not None:
            try:
                quick_report = self.fit_predict(
                    user_measurements, garment_measurements, fit_preference
                )
                print("[Fit Predictor] 핏 판정 완료")
                self.print_fit_report(quick_report)
                self._print_size_info(quick_report, garment_measurements)
            except Exception as _fit_exc:
                print(f"[Fit Predictor] 핏 판정 건너뜀: {_fit_exc}")

        self._ensure_preprocessing_on_gpu()

        src_image = Image.open(src_image_path).convert("RGB")
        ref_image = Image.open(ref_image_path).convert("RGB")
        src_image = resize_and_center(src_image, 768, 1024)
        ref_image = resize_and_center(ref_image, 768, 1024)

        parsing_map, _ = self.parsing(src_image.resize((768, 1024)))
        parsing_map = np.array(parsing_map)

        upper_body_mask_np = np.isin(parsing_map, [4]).astype(np.uint8)
        arms_mask_np = np.isin(parsing_map, [14, 15]).astype(np.uint8)
        hands_mask_np = np.isin(parsing_map, [20, 21]).astype(np.uint8)
        inpaint_mask_np = upper_body_mask_np | arms_mask_np | hands_mask_np

        kernel = np.ones((10, 10), np.uint8)
        inpaint_mask_np_dilated = cv2.dilate(
            inpaint_mask_np.astype(np.uint8), kernel, iterations=1
        )
        inpaint_mask_img = Image.fromarray(inpaint_mask_np_dilated * 255)

        fit_requested = self._fit_mode_requested(
            user_measurements=user_measurements,
            garment_measurements=garment_measurements,
            control_type=control_type,
        )
        fit_strategy = (fit_strategy or "geometry").lower()

        if not np.any(inpaint_mask_np):
            empty_mask = Image.fromarray(np.zeros((1024, 768), dtype=np.uint8))
            if output_path:
                src_image.save(output_path)
            return self._format_outputs(
                generated_image=src_image,
                mask=empty_mask,
                densepose=empty_mask,
                agnostic_image=src_image,
                fit_outputs=None,
                return_fit_debug=(return_fit_debug or fit_requested),
            )

        agnostic_image = self.generate_skin(
            src_image=src_image,
            inpaint_mask_img=inpaint_mask_img,
            step=step,
            seed=seed,
        )

        self._free_gpu()
        self._ensure_preprocessing_on_gpu()

        if control_type == "virtual_tryon":
            garment_mapping = {
                "dresses": "overall",
                "upper_body": "upper",
                "lower_body": "lower",
                "short_sleeve": "short_sleeve",
                "shorts": "shorts",
            }
            garment_type_hd = garment_mapping.get(vt_garment_type, "upper")
            mask = self.mask_predictor(agnostic_image, garment_type_hd)["mask"]
            if src_mask_path:
                mask.save(src_mask_path)
        else:
            agnostic_np = np.array(agnostic_image)
            mask = Image.fromarray(np.ones_like(agnostic_np, dtype=np.uint8) * 255)

        agnostic_np = np.array(agnostic_image)
        if vt_model_type == "viton_hd":
            seg = self.densepose_predictor.predict_seg(agnostic_np)[:, :, ::-1]
        else:
            iuv = self.densepose_predictor.predict_iuv(agnostic_np)
            seg = np.concatenate([iuv[:, :, :1]] * 3, axis=-1)
        densepose = Image.fromarray(seg)

        fit_outputs = None
        refined_mask = mask
        inference_ref_image = ref_image
        masked_image_mask = refined_mask
        if fit_requested:
            try:
                if fit_strategy == "geometry":
                    fit_outputs = self._run_fit_aware_geometry(
                        base_mask=mask,
                        densepose=densepose,
                        parsing_map=parsing_map,
                        ref_image=ref_image,
                        user_measurements=user_measurements,
                        garment_measurements=garment_measurements,
                        preferred_fit=preferred_fit,
                        vt_garment_type=vt_garment_type,
                    )
                else:
                    fit_outputs = self._run_fit_aware_refinement(
                        base_mask=mask,
                        densepose=densepose,
                        user_measurements=user_measurements,
                        garment_measurements=garment_measurements,
                        preferred_fit=preferred_fit,
                        vt_garment_type=vt_garment_type,
                    )
                refined_mask = fit_outputs["refined_mask"]
                inference_ref_image = fit_outputs.get("warped_garment_image", ref_image)
                masked_image_mask = fit_outputs.get("masked_image_mask", mask)
                # Apply fit-aware mask morphology
                refined_mask = self._adjust_mask_for_fit(
                    refined_mask, fit_outputs.get("fit_report"),
                )
            except Exception as exc:
                log.exception(
                    "[capvton] ❌ 핏 추론 실패 (fit_strategy=%s) — 피팅 이미지는 생성되지만 fitReport는 null로 반환됩니다.",
                    fit_strategy,
                )
                fit_outputs = {
                    "fit_report": None,
                    "refined_mask": mask,
                    "base_mask": mask,
                    "warped_garment_image": ref_image,
                    "warped_garment_mask": None,
                    "layout_hint": None,
                    "target_mask": None,
                    "sdf_map": None,
                    "layout_cond": None,
                    "debug": {
                        "base_mask": mask,
                        "layout_source": "fallback",
                        "fit_error": str(exc),
                        "fit_strategy": fit_strategy,
                    },
                }
                inference_ref_image = ref_image
                masked_image_mask = mask

        self._offload_preprocessing()

        transform = LeffaTransform()
        data = {
            "src_image": [agnostic_image],
            "ref_image": [inference_ref_image],
            "mask": [refined_mask],
            "masked_image_mask": [masked_image_mask if fit_outputs else refined_mask],
            "densepose": [densepose],
        }
        data = transform(data)

        inference = self._get_vt_inference(vt_model_type)

        garment_prompt, negative_prompt = self._build_fit_prompts(fit_outputs)

        extra_conditions = fit_outputs.get("layout_cond") if fit_outputs else None
        if isinstance(extra_conditions, torch.Tensor):
            extra_conditions = {"layout_cond": extra_conditions}
        # Inject fit tightness + warped garment into pipeline conditioning
        if fit_outputs and fit_outputs.get("fit_report") is not None:
            avg_tightness = self._compute_avg_tightness(fit_outputs["fit_report"])
            if extra_conditions is None:
                extra_conditions = {}
            extra_conditions["fit_tightness"] = avg_tightness

            # KEY FIX: pass the warped garment tensors so the pipeline can
            # blend them into masked_image, overriding body-shape dependency.
            warped_img_tensor = fit_outputs.get("warped_garment_image_tensor")
            warped_msk_tensor = fit_outputs.get("warped_garment_mask_tensor")
            if warped_img_tensor is not None and warped_msk_tensor is not None:
                # Normalise from [0,1] → [-1,1] to match src_image scale in pipeline
                extra_conditions["warped_garment"] = warped_img_tensor * 2.0 - 1.0
                extra_conditions["warped_garment_mask"] = warped_msk_tensor

        result = inference(
            data,
            ref_acceleration=ref_acceleration,
            num_inference_steps=step,
            cross_attention_kwargs=cross_attention_kwargs,
            seed=seed,
            repaint=vt_repaint,
            prompt=garment_prompt,
            negative_prompt=negative_prompt,
            extra_conditions=extra_conditions,
        )

        gen_image = result["generated_image"][0]

        # Apply fit-aware post-processing visual effects
        if fit_outputs and fit_outputs.get("fit_report") is not None:
            gen_image = self._apply_fit_post_processing(
                gen_image, agnostic_image, densepose, refined_mask,
                fit_outputs["fit_report"],
            )

        if output_path:
            gen_image.save(output_path)
            output_dir = os.path.dirname(output_path) or "."
            gt_path = os.path.join(output_dir, "ground_truth.png")
            pred_path = os.path.join(output_dir, "prediction.png")
            ref_image.save(gt_path)
            gen_image.save(pred_path)
            print(f"Ground Truth saved at: {gt_path}")
            print(f"Prediction saved at: {pred_path}")

            if fit_outputs and fit_outputs.get("fit_report") is not None:
                save_fit_artifacts(
                    output_dir=output_dir,
                    fit_report=fit_outputs["fit_report"],
                    target_mask=fit_outputs.get("target_mask_tensor"),
                    sdf_map=fit_outputs.get("sdf_map_tensor"),
                    refined_mask=fit_outputs.get("refined_mask_tensor"),
                    warped_garment_image=fit_outputs.get("warped_garment_image"),
                    warped_garment_mask=fit_outputs.get("warped_garment_mask_tensor"),
                    adapted_garment_image=fit_outputs.get("adapted_garment_image_tensor"),
                    adapted_garment_mask=fit_outputs.get("adapted_garment_mask_tensor"),
                    vmg_mask=fit_outputs.get("vmg_mask_tensor"),
                    vmg_grid=fit_outputs.get("vmg_grid_tensor"),
                    layout_hint=fit_outputs.get("layout_hint_tensor"),
                    base_mask=fit_outputs.get("base_mask_tensor"),
                )

        return self._format_outputs(
            generated_image=gen_image,
            mask=mask,
            densepose=densepose,
            agnostic_image=agnostic_image,
            fit_outputs=fit_outputs,
            return_fit_debug=(return_fit_debug or fit_requested),
        )

    # ------------------------------------------------------------------
    # Size recommendation API
    # ------------------------------------------------------------------
    def recommend_size(
        self,
        user_measurements,
        size_chart=None,
        category: str = "upper_body",
        preferred_fit: str = "regular",
    ):
        """Return a SizeRecommendation without generating any images.

        Args:
            user_measurements: dict or UserMeasurements with body measurements.
            size_chart: dict mapping size_label → garment measurements.
                        If *None*, uses built-in standard Korean/unisex chart.
            category: garment category (used only when *size_chart* is None).
            preferred_fit: "slim", "regular", "relaxed", or "oversized".

        Returns:
            SizeRecommendation dataclass (call ``.to_dict()`` for JSON-safe output).
        """
        if size_chart is not None:
            return self.size_recommender.recommend(
                user=user_measurements,
                size_chart=size_chart,
                preferred_fit=preferred_fit,
            )
        return self.size_recommender.recommend_from_standard_sizes(
            user=user_measurements,
            category=category,
            preferred_fit=preferred_fit,
        )

    def predict_with_size_recommendation(
        self,
        src_image_path,
        ref_image_path,
        user_measurements,
        size_chart=None,
        category: str = "upper_body",
        preferred_fit: str = "regular",
        output_dir: str = None,
        top_k: int = 3,
        **predict_kwargs,
    ):
        """Recommend sizes AND generate try-on images for the top-k sizes.

        Returns:
            dict with "recommendation" (SizeRecommendation) and
            "tryon_results" mapping size_label → capvton_predict output.
        """
        rec = self.recommend_size(
            user_measurements=user_measurements,
            size_chart=size_chart,
            category=category,
            preferred_fit=preferred_fit,
        )

        tryon_results = {}
        for score in rec.all_sizes[:top_k]:
            size_label = score.size_label
            garment_m = score.fit_report.to_dict()

            size_output_path = None
            if output_dir:
                size_output_path = os.path.join(
                    output_dir, f"tryon_{size_label}.png",
                )

            result = self.capvton_predict(
                src_image_path=src_image_path,
                ref_image_path=ref_image_path,
                control_type="virtual_tryon",
                output_path=size_output_path,
                user_measurements=user_measurements,
                garment_measurements=score.fit_report.to_dict().get(
                    "garment_measurements",
                    self._size_chart_entry(size_chart, size_label, category),
                ),
                preferred_fit=preferred_fit,
                return_fit_debug=True,
                vt_garment_type=category,
                **predict_kwargs,
            )
            tryon_results[size_label] = result

        return {
            "recommendation": rec,
            "tryon_results": tryon_results,
        }

    @staticmethod
    def _size_chart_entry(size_chart, size_label, category):
        """Extract garment measurements for a specific size."""
        if size_chart and size_label in size_chart:
            return size_chart[size_label]
        from capvton.fit.size_recommender import STANDARD_SIZE_CHARTS
        chart = STANDARD_SIZE_CHARTS.get(category, {})
        return chart.get(size_label, {})
