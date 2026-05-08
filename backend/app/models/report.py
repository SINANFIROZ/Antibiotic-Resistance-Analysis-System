import enum

from sqlalchemy import Enum, ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base
from app.models.base_mixins import TimestampMixin, UUIDPrimaryKeyMixin


class ReportStatus(str, enum.Enum):
    GENERATED = 'generated'
    DELIVERED = 'delivered'
    FAILED = 'failed'


class Report(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = 'reports'

    prediction_id: Mapped[str] = mapped_column(ForeignKey('predictions.id'))
    created_by_id: Mapped[str] = mapped_column(ForeignKey('users.id'))
    file_path: Mapped[str] = mapped_column(String(500))
    status: Mapped[ReportStatus] = mapped_column(Enum(ReportStatus), default=ReportStatus.GENERATED)
    emailed_to: Mapped[str | None] = mapped_column(String(255), nullable=True)

    prediction = relationship('Prediction', back_populates='reports')
    created_by = relationship('User', back_populates='reports')
