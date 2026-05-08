from datetime import date

from sqlalchemy import Boolean, Date, Float, ForeignKey, JSON, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base
from app.models.base_mixins import TimestampMixin, UUIDPrimaryKeyMixin


class ResistanceHistory(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = 'resistance_history'

    microbe_id: Mapped[str] = mapped_column(ForeignKey('microbes.id'))
    antibiotic_id: Mapped[str] = mapped_column(ForeignKey('antibiotics.id'))
    region: Mapped[str | None] = mapped_column(String(120), nullable=True)
    facility_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    sample_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    result_resistant: Mapped[bool] = mapped_column(Boolean, default=False)
    mic_value: Mapped[float | None] = mapped_column(Float, nullable=True)
    genomic_markers: Mapped[list[str] | None] = mapped_column(JSON, nullable=True)

    microbe = relationship('Microbe', back_populates='history')
    antibiotic = relationship('Antibiotic', back_populates='history')
