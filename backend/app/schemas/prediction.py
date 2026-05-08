from datetime import datetime

from pydantic import BaseModel, Field

from app.models.prediction import PredictionLabel


class PredictionRequest(BaseModel):
    patient_id: str | None = None
    microbe_id: str
    antibiotic_id: str
    facility_name: str | None = None
    region: str | None = None
    mic_value: float | None = None
    genomic_markers: list[str] = Field(default_factory=list)


class PredictionRead(BaseModel):
    id: str
    created_at: datetime
    prediction_label: PredictionLabel
    resistant_probability: float
    confidence_score: float
    explanation_text: str
    recommended_alternatives: list[str]
    shap_summary: list[dict]
    model_version: str
    patient_id: str | None
    microbe_id: str
    antibiotic_id: str
    requested_by_id: str
