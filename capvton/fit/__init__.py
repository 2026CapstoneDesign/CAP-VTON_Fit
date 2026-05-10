"""
Fit-Aware Virtual Try-On — capvton.fit package
"""

from capvton.fit.schema import (
    UserMeasurements,
    GarmentMeasurements,
    EaseVector,
    FitReport,
    FitClass,
    Gender,
    GarmentCategory,
    GarmentSuperCategory,
)
from capvton.fit.fit_predictor_rule import RuleBasedFitPredictor
from capvton.fit.film_adapter import FitEmbeddingEncoder
from capvton.fit.layout_generator import FitLayoutGenerator
from capvton.fit.fit_modules import (
    FitPredictor,
    FitStateEncoder,
    BodyAnchorEncoder,
    GarmentGeometryGenerator,
    HeuristicGarmentGeometryGenerator,
    FitAwareGarmentAdapter,
    VirtualMeasurementGarmentBuilder,
    HeuristicLayoutRefiner,
    coerce_user_measurements,
    coerce_garment_measurements,
    resolve_category,
    run_garment_geometry,
    run_fit_adapter,
    run_layout_generation,
)
from capvton.fit.size_recommender import SizeRecommender
