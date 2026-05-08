import enum

from sqlalchemy import Enum, Float, ForeignKey, JSON, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base
from app.models.base_mixins import TimestampMixin, UUIDPrimaryKeyMixin


class PredictionLabel(str, enum.Enum):
    RESISTANT = 'resistant'
    SUSCEPTIBLE = 'susceptible'


class Prediction(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = 'predictions'

    patient_id: Mapped[str | None] = mapped_column(ForeignKey('patients.id'), nullable=True)
    microbe_id: Mapped[str] = mapped_column(ForeignKey('microbes.id'))
    antibiotic_id: Mapped[str] = mapped_column(ForeignKey('antibiotics.id'))
    requested_by_id: Mapped[str] = mapped_column(ForeignKey('users.id'))
    prediction_label: Mapped[PredictionLabel] = mapped_column(Enum(PredictionLabel), nullable=False)
    resistant_probability: Mapped[float] = mapped_column(Float, nullable=False)
    confidence_score: Mapped[float] = mapped_column(Float, nullable=False)
    explanation_text: Mapped[str] = mapped_column(Text, default='')
    recommended_alternatives: Mapped[list[str]] = mapped_column(JSON, default=list)
    shap_summary: Mapped[list[dict]] = mapped_column(JSON, default=list)
    model_version: Mapped[str] = mapped_column(String(100), default='legacy-xgb-v1')

    patient = relationship('Patient', back_populates='predictions')
    microbe = relationship('Microbe', back_populates='predictions')
    antibiotic = relationship('Antibiotic', back_populates='predictions')
    requested_by = relationship('User', back_populates='predictions')
    reports = relationship('Report', back_populates='prediction')
