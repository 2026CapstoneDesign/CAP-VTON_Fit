from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


STANDARD_SIZE_CHARTS: Dict[str, Dict[str, Dict]] = {
    "upper_body": {
        "XS":  {"length": 65, "chest_width": 46, "shoulder": 43, "sleeve_length": 59, "waist_width": 43, "category": "tshirt"},
        "S":   {"length": 67, "chest_width": 48, "shoulder": 44, "sleeve_length": 60, "waist_width": 45, "category": "tshirt"},
        "M":   {"length": 69, "chest_width": 51, "shoulder": 46, "sleeve_length": 61, "waist_width": 48, "category": "tshirt"},
        "L":   {"length": 71, "chest_width": 54, "shoulder": 48, "sleeve_length": 62, "waist_width": 51, "category": "tshirt"},
        "XL":  {"length": 73, "chest_width": 57, "shoulder": 50, "sleeve_length": 63, "waist_width": 54, "category": "tshirt"},
        "XXL": {"length": 75, "chest_width": 61, "shoulder": 53, "sleeve_length": 64, "waist_width": 58, "category": "tshirt"},
    },
    "lower_body": {
        "XS":  {"length": 96,  "waist_width": 33, "hip_width": 44, "thigh_width": 26, "inseam": 70, "category": "pants"},
        "S":   {"length": 97,  "waist_width": 35, "hip_width": 46, "thigh_width": 27, "inseam": 71, "category": "pants"},
        "M":   {"length": 98,  "waist_width": 38, "hip_width": 49, "thigh_width": 29, "inseam": 72, "category": "pants"},
        "L":   {"length": 99,  "waist_width": 41, "hip_width": 52, "thigh_width": 31, "inseam": 73, "category": "pants"},
        "XL":  {"length": 100, "waist_width": 44, "hip_width": 55, "thigh_width": 33, "inseam": 74, "category": "pants"},
        "XXL": {"length": 101, "waist_width": 48, "hip_width": 59, "thigh_width": 36, "inseam": 75, "category": "pants"},
    },
    "dresses": {
        "XS":  {"length": 95,  "chest_width": 44, "waist_width": 37, "hip_width": 46, "category": "dress"},
        "S":   {"length": 97,  "chest_width": 46, "waist_width": 39, "hip_width": 48, "category": "dress"},
        "M":   {"length": 99,  "chest_width": 49, "waist_width": 42, "hip_width": 51, "category": "dress"},
        "L":   {"length": 101, "chest_width": 52, "waist_width": 45, "hip_width": 54, "category": "dress"},
        "XL":  {"length": 103, "chest_width": 55, "waist_width": 48, "hip_width": 57, "category": "dress"},
        "XXL": {"length": 105, "chest_width": 59, "waist_width": 52, "hip_width": 61, "category": "dress"},
    },
}


@dataclass
class SizeScore:
    size_label: str
    fit_report: Any
    score: float


@dataclass
class SizeRecommendation:
    best_size: str
    all_sizes: List[SizeScore]
    category: str = "upper_body"

    def to_dict(self) -> dict:
        return {
            "best_size": self.best_size,
            "category": self.category,
            "all_sizes": [
                {
                    "size_label": s.size_label,
                    "score": s.score,
                    "fit_report": s.fit_report.to_dict() if s.fit_report else None,
                }
                for s in self.all_sizes
            ],
        }


class SizeRecommender:
    def __init__(self, fit_predictor):
        self.fit_predictor = fit_predictor

    def recommend(
        self,
        user: Any,
        size_chart: Dict[str, Any],
        preferred_fit: str = "regular",
    ) -> SizeRecommendation:
        from capvton.fit.fit_modules import coerce_user_measurements, coerce_garment_measurements

        if isinstance(user, dict):
            user = coerce_user_measurements(user)

        scores: List[SizeScore] = []
        for label, spec in size_chart.items():
            if label.startswith("_"):
                continue
            garment = coerce_garment_measurements(spec) if isinstance(spec, dict) else spec
            garment.size_label = label
            report = self.fit_predictor.predict(user, garment, preferred_fit=preferred_fit)
            scores.append(SizeScore(size_label=label, fit_report=report, score=report.overall_score))

        scores.sort(key=lambda s: -s.score)
        best = scores[0].size_label if scores else "M"
        return SizeRecommendation(best_size=best, all_sizes=scores)

    def recommend_from_standard_sizes(
        self,
        user: Any,
        category: str = "upper_body",
        preferred_fit: str = "regular",
    ) -> SizeRecommendation:
        from capvton.fit.fit_modules import coerce_user_measurements, coerce_garment_measurements

        if isinstance(user, dict):
            user = coerce_user_measurements(user)

        chart_raw = STANDARD_SIZE_CHARTS.get(category, STANDARD_SIZE_CHARTS["upper_body"])
        scores: List[SizeScore] = []
        for label, spec in chart_raw.items():
            garment = coerce_garment_measurements({**spec, "category": spec.get("category", category)})
            garment.size_label = label
            report = self.fit_predictor.predict(user, garment, preferred_fit=preferred_fit)
            scores.append(SizeScore(size_label=label, fit_report=report, score=report.overall_score))

        scores.sort(key=lambda s: -s.score)
        best = scores[0].size_label if scores else "M"
        return SizeRecommendation(best_size=best, all_sizes=scores, category=category)
