from sqlalchemy import ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base
from app.models.base_mixins import TimestampMixin, UUIDPrimaryKeyMixin


class UploadedDataset(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = 'uploaded_datasets'

    filename: Mapped[str] = mapped_column(String(255))
    source: Mapped[str] = mapped_column(String(120), default='manual_upload')
    schema_version: Mapped[str] = mapped_column(String(50), default='v1')
    row_count: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[str] = mapped_column(String(50), default='uploaded')
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    uploaded_by_id: Mapped[str] = mapped_column(ForeignKey('users.id'))

    uploaded_by = relationship('User', back_populates='datasets')
