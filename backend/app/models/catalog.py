from sqlalchemy import Float, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base
from app.models.base_mixins import TimestampMixin, UUIDPrimaryKeyMixin


class Microbe(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = 'microbes'

    name: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    taxonomy_group: Mapped[str | None] = mapped_column(String(120), nullable=True)
    genome_reference: Mapped[str | None] = mapped_column(String(255), nullable=True)
    baseline_resistance_rate: Mapped[float] = mapped_column(Float, default=0.5)

    predictions = relationship('Prediction', back_populates='microbe')
    history = relationship('ResistanceHistory', back_populates='microbe')


class Antibiotic(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = 'antibiotics'

    name: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    antibiotic_class: Mapped[str | None] = mapped_column(String(120), nullable=True)
    who_awarea_category: Mapped[str | None] = mapped_column(String(50), nullable=True)

    predictions = relationship('Prediction', back_populates='antibiotic')
    history = relationship('ResistanceHistory', back_populates='antibiotic')
