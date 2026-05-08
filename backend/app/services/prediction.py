import json
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.models.catalog import Antibiotic, Microbe
from app.models.prediction import Prediction, PredictionLabel
from app.models.resistance_history import ResistanceHistory
from app.models.user import User
from app.schemas.prediction import PredictionRequest
from app.services.audit import create_audit_log

settings = get_settings()
LEGACY_EXPLANATION_TEMPLATE = (
    'Legacy ensemble scored {microbe_name} against {antibiotic_name} with a resistant probability of '
    '{probability:.2%}. Confidence reflects distance from the decision threshold and available historical signal.'
)
FALLBACK_EXPLANATION_TEMPLATE = (
    'Fallback heuristic used for {microbe_name} and {antibiotic_name} because {reason}. '
    'Probability is derived from baseline resistance prevalence and should be validated with a trained registry model.'
)


@dataclass
class PredictionResult:
    label: PredictionLabel
    resistant_probability: float
    confidence_score: float
    explanation_text: str
    alternatives: list[str]
    shap_summary: list[dict]
    model_version: str


class LegacyModelGateway:
    def __init__(self) -> None:
        self._joblib = None
        self._numpy = None

    def _load_dependencies(self) -> bool:
        try:
            import joblib
            import numpy as np
        except ImportError:
            return False
        self._joblib = joblib
        self._numpy = np
        return True

    def _locate_model_dir(self):
        for model_dir in settings.legacy_model_dirs:
            if model_dir.exists():
                return model_dir
        return None

    def predict(self, microbe_name: str, antibiotic_name: str, baseline_rate: float) -> PredictionResult:
        if not self._load_dependencies():
            return self._fallback_prediction(microbe_name, antibiotic_name, baseline_rate, 'optional-ml-deps-missing')

        model_dir = self._locate_model_dir()
        if model_dir is None:
            return self._fallback_prediction(microbe_name, antibiotic_name, baseline_rate, 'legacy-models-missing')

        encoder_path = model_dir / 'species_encoder.pkl'
        model_path = model_dir / f'model_{antibiotic_name}.pkl'
        alt_path = model_dir / 'alternative_antibiotics.json'
        if not encoder_path.exists() or not model_path.exists():
            return self._fallback_prediction(microbe_name, antibiotic_name, baseline_rate, 'model-artifact-missing')

        encoder = self._joblib.load(encoder_path)
        model = self._joblib.load(model_path)
        encoded_value = encoder.transform([microbe_name])[0]
        features = self._numpy.array([[encoded_value]])
        prediction = int(model.predict(features)[0])
        probability = float(model.predict_proba(features)[0][1])
        alternatives: list[str] = []
        if alt_path.exists():
            with alt_path.open() as handle:
                alt_map = json.load(handle)
            alternatives = alt_map.get(microbe_name, {}).get(antibiotic_name, [])[:5]

        label = PredictionLabel.RESISTANT if prediction == 1 else PredictionLabel.SUSCEPTIBLE
        confidence = round(abs(probability - 0.5) * 2, 4)
        explanation = LEGACY_EXPLANATION_TEMPLATE.format(
            microbe_name=microbe_name,
            antibiotic_name=antibiotic_name,
            probability=probability,
        )
        shap_summary = [
            {
                'feature': 'species',
                'impact': round(probability - baseline_rate, 4),
                'direction': 'increase' if probability >= baseline_rate else 'decrease',
            }
        ]
        return PredictionResult(
            label=label,
            resistant_probability=probability,
            confidence_score=confidence,
            explanation_text=explanation,
            alternatives=alternatives,
            shap_summary=shap_summary,
            model_version='legacy-xgb-v1',
        )

    def _fallback_prediction(self, microbe_name: str, antibiotic_name: str, baseline_rate: float, reason: str) -> PredictionResult:
        probability = max(0.15, min(0.85, baseline_rate))
        label = PredictionLabel.RESISTANT if probability >= 0.5 else PredictionLabel.SUSCEPTIBLE
        return PredictionResult(
            label=label,
            resistant_probability=probability,
            confidence_score=0.42,
            explanation_text=(
                FALLBACK_EXPLANATION_TEMPLATE.format(
                    microbe_name=microbe_name,
                    antibiotic_name=antibiotic_name,
                    reason=reason,
                )
            ),
            alternatives=[],
            shap_summary=[{'feature': 'baseline_resistance_rate', 'impact': baseline_rate, 'direction': 'increase'}],
            model_version='heuristic-baseline-v1',
        )


async def create_prediction(db: AsyncSession, payload: PredictionRequest, requested_by: User) -> Prediction:
    microbe = await db.scalar(select(Microbe).where(Microbe.id == payload.microbe_id))
    antibiotic = await db.scalar(select(Antibiotic).where(Antibiotic.id == payload.antibiotic_id))
    if microbe is None or antibiotic is None:
        raise ValueError('Invalid microbe or antibiotic reference')

    history_stmt = select(ResistanceHistory).where(
        ResistanceHistory.microbe_id == microbe.id,
        ResistanceHistory.antibiotic_id == antibiotic.id,
    )
    history = (await db.scalars(history_stmt)).all()
    baseline_rate = microbe.baseline_resistance_rate
    if history:
        baseline_rate = sum(1 for item in history if item.result_resistant) / len(history)

    gateway = LegacyModelGateway()
    result = gateway.predict(microbe.name, antibiotic.name, baseline_rate)

    prediction = Prediction(
        patient_id=payload.patient_id,
        microbe_id=microbe.id,
        antibiotic_id=antibiotic.id,
        requested_by_id=requested_by.id,
        prediction_label=result.label,
        resistant_probability=result.resistant_probability,
        confidence_score=result.confidence_score,
        explanation_text=result.explanation_text,
        recommended_alternatives=result.alternatives,
        shap_summary=result.shap_summary,
        model_version=result.model_version,
    )
    db.add(prediction)
    await db.flush()
    await create_audit_log(
        db,
        'prediction.created',
        'prediction',
        prediction.id,
        requested_by.id,
        {
            'microbe_id': microbe.id,
            'antibiotic_id': antibiotic.id,
            'label': result.label.value,
            'probability': result.resistant_probability,
        },
    )
    await db.commit()
    await db.refresh(prediction)
    return prediction
